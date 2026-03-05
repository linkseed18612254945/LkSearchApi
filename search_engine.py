"""
搜索引擎核心 — 整合 Level 1 + 懒计算向量重排序

排序模式：
  hybrid        — BM25 × 多信号加权（时效+权威+质量）
  hybrid_vector — BM25 召回 → 懒计算向量重排序（缓存优先，按需调 API）
  relevance     — 纯 BM25
  date          — 纯时间倒序

hybrid_vector 流程：
  ① BM25 + Level1 多信号召回 RERANK_RECALL_SIZE 篇
  ② 查询词向量：SQLite 缓存 → 未命中调 API
  ③ 文档向量：批量查 SQLite → 只对缺失的调 API → 写回缓存
  ④ 余弦相似度打分，与 BM25 分加权合并
  ⑤ 重新排序，取用户请求的页
"""

import math
import time
import logging
from typing import Optional

from app.core.es_client import es_client
from app.core.config import settings
from app.services.reranker import lazy_reranker
from app.models.search import (
    SearchRequest, SearchResponse, SearchResultItem,
    ScoreBreakdown, SortMode,
)

logger = logging.getLogger(__name__)


class SearchEngine:

    F_TITLE    = settings.FIELD_TITLE
    F_CONTENT  = settings.FIELD_CONTENT
    F_DATE     = settings.FIELD_DATE
    F_SOURCE   = settings.FIELD_SOURCE
    F_URL      = settings.FIELD_URL
    F_SRC_RANK = settings.FIELD_SOURCE_RANK
    F_CTR_LEN  = settings.FIELD_CONTENT_LENGTH
    F_CLICKS   = settings.FIELD_CLICK_COUNT

    # ── 主入口 ────────────────────────────────────────────────────

    async def search(self, req: SearchRequest) -> SearchResponse:
        start   = time.time()
        indices = self._resolve_indices(req.indices)

        if req.sort == SortMode.hybrid_vector:
            results, total = await self._search_with_rerank(req, indices)
        else:
            results, total = await self._search_bm25(req, indices)

        took_ms     = int((time.time() - start) * 1000)
        total_pages = math.ceil(total / req.page_size) if total > 0 else 0

        return SearchResponse(
            query=req.query,
            sort_mode=req.sort.value,
            total=total,
            page=req.page,
            page_size=req.page_size,
            total_pages=total_pages,
            took_ms=took_ms,
            results=results,
        )

    # ── hybrid_vector：BM25 召回 + 懒计算重排序 ──────────────────

    async def _search_with_rerank(
        self, req: SearchRequest, indices: str
    ) -> tuple[list[SearchResultItem], int]:
        """
        第一步：用 Level1 多信号 BM25 召回更多候选（RERANK_RECALL_SIZE 篇）
        第二步：懒计算向量重排序
        第三步：手动分页截取
        """
        # 召回足够多的候选，才能让重排序有意义
        # 例如用户要第 2 页每页 10 条，需要召回至少 max(50, 20) 篇
        recall_size = max(
            settings.RERANK_RECALL_SIZE,
            req.page * req.page_size,
        )

        dsl = self._dsl_level1_multisignal(req)
        raw = await es_client.search(
            index=indices,
            body=dsl,
            size=recall_size,  # 不用 from_，先召回全部再手动分页
        )
        all_hits = raw["hits"]["hits"]
        total    = raw["hits"]["total"]["value"]

        if not all_hits:
            return [], total

        # 向量重排序（内部处理缓存）
        reranked = await lazy_reranker.rerank(
            query=req.query,
            hits=all_hits,
            index=indices.split(",")[0],  # 多索引时用第一个作为缓存 key 前缀
            debug=req.debug_scores,
        )

        # 手动分页
        page_start = (req.page - 1) * req.page_size
        page_hits  = reranked[page_start: page_start + req.page_size]

        results = self._parse_hits(page_hits, req, mode="rerank")
        # total 用召回总量（表示有多少文章匹配查询），不是 recall_size
        return results, total

    # ── BM25 搜索（hybrid / relevance / date）────────────────────

    async def _search_bm25(
        self, req: SearchRequest, indices: str
    ) -> tuple[list[SearchResultItem], int]:
        dsl = self._build_bm25_dsl(req)
        raw = await es_client.search(
            index=indices,
            body=dsl,
            from_=(req.page - 1) * req.page_size,
            size=req.page_size,
        )
        hits  = raw["hits"]["hits"]
        total = raw["hits"]["total"]["value"]
        return self._parse_hits(hits, req), total

    def _build_bm25_dsl(self, req: SearchRequest) -> dict:
        if req.sort == SortMode.hybrid:
            return self._dsl_level1_multisignal(req)
        elif req.sort == SortMode.relevance:
            return {"query": self._base_query(req), "highlight": self._highlight_cfg(req)}
        else:  # date
            return {
                "query": self._base_query(req),
                "sort":  [{self.F_DATE: "desc"}, "_score"],
                "highlight": self._highlight_cfg(req),
            }

    # ── Level 1 多信号 DSL ────────────────────────────────────────

    def _dsl_level1_multisignal(self, req: SearchRequest) -> dict:
        """
        BM25 × (时效分 + 权威分 + 质量分 + 热度分)

        score_mode="sum"  → 4 个 function 分数先相加
        boost_mode="multiply" → 再乘以 BM25 原始分

        这样做的效果：
          一篇文章要排在最前，必须同时满足"相关 AND 新 AND 权威"
          而不是某一项极高就能压过其他
        """
        return {
            "query": {
                "function_score": {
                    "query": self._base_query(req),
                    "functions": [
                        # ① 时效：高斯衰减，30天前降到 0.5
                        {
                            "gauss": {
                                self.F_DATE: {
                                    "origin": "now",
                                    "scale":  settings.FRESHNESS_SCALE,
                                    "decay":  settings.FRESHNESS_DECAY,
                                }
                            },
                            "weight": settings.WEIGHT_FRESHNESS,
                        },
                        # ② 来源权威：直接读字段值，缺失默认 0.5
                        {
                            "field_value_factor": {
                                "field":    self.F_SRC_RANK,
                                "factor":   1.0,
                                "modifier": "none",
                                "missing":  0.5,
                            },
                            "weight": settings.WEIGHT_SOURCE_RANK,
                        },
                        # ③ 内容质量：log(正文长度)，压缩超长文章优势
                        {
                            "field_value_factor": {
                                "field":    self.F_CTR_LEN,
                                "factor":   0.1,
                                "modifier": "log1p",
                                "missing":  1.0,
                            },
                            "weight": settings.WEIGHT_QUALITY,
                        },
                        # ④ 点击热度：log(点击数)，没有数据给 0
                        {
                            "field_value_factor": {
                                "field":    self.F_CLICKS,
                                "factor":   0.1,
                                "modifier": "log1p",
                                "missing":  0.0,
                            },
                            "weight": settings.WEIGHT_HOTNESS,
                        },
                    ],
                    "score_mode": "sum",
                    "boost_mode": "multiply",
                }
            },
            "highlight": self._highlight_cfg(req),
        }

    # ── 公共辅助 ──────────────────────────────────────────────────

    def _resolve_indices(self, param: Optional[str]) -> str:
        if not param:
            return settings.DEFAULT_INDEX
        return ",".join(i.strip() for i in param.split(","))

    def _base_query(self, req: SearchRequest) -> dict:
        must = [{
            "multi_match": {
                "query":   req.query,
                "fields":  [f"{self.F_TITLE}^3", f"{self.F_CONTENT}^1"],
                "type":    "best_fields",
                "operator": "or",
                "minimum_should_match": "60%",
                "fuzziness": "AUTO",
            }
        }]
        return {"bool": {"must": must, "filter": self._date_filter(req)}}

    def _date_filter(self, req: SearchRequest) -> list[dict]:
        if not req.date_from and not req.date_to:
            return []
        r = {}
        if req.date_from:
            r["gte"] = req.date_from.isoformat()
        if req.date_to:
            r["lte"] = req.date_to.isoformat()
        return [{"range": {self.F_DATE: r}}]

    def _highlight_cfg(self, req: SearchRequest) -> dict:
        if not req.include_highlights:
            return {}
        return {
            "pre_tags":  ["<em>"],
            "post_tags": ["</em>"],
            "fields": {
                self.F_TITLE:   {"number_of_fragments": 1},
                self.F_CONTENT: {"number_of_fragments": 3, "fragment_size": 150},
            },
        }

    def _parse_hits(
        self, hits: list[dict], req: SearchRequest, mode: str = "bm25"
    ) -> list[SearchResultItem]:
        results = []
        for hit in hits:
            src       = hit.get("_source", {})
            highlight = hit.get("highlight", {})
            content   = src.get(self.F_CONTENT, "") or ""

            breakdown = None
            if req.debug_scores:
                breakdown = ScoreBreakdown(
                    final_score=round(hit.get("_score") or 0, 6),
                    bm25_score=round(hit.get("_original_score") or hit.get("_score") or 0, 4),
                    vector_score=round(hit.get("_vector_score") or 0, 4) if mode == "rerank" else None,
                    source_rank=src.get(self.F_SRC_RANK),
                )

            results.append(SearchResultItem(
                id=hit["_id"],
                title=src.get(self.F_TITLE, "（无标题）"),
                url=src.get(self.F_URL),
                source=src.get(self.F_SOURCE),
                published_at=src.get(self.F_DATE),
                summary=content[:200] + ("..." if len(content) > 200 else ""),
                highlights={
                    "title":   highlight.get(self.F_TITLE, []),
                    "content": highlight.get(self.F_CONTENT, []),
                } if highlight else None,
                score=round(hit.get("_score") or 0, 6),
                score_breakdown=breakdown,
            ))
        return results


search_engine = SearchEngine()
