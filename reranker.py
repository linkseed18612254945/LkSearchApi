"""
懒计算重排序服务（Lazy Reranker）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心思想：
  不预先计算所有文档的向量，而是"用到才算，算完就存"

工作流：
  1. BM25 召回 N 篇文章（已有的搜索结果）
  2. 批量查 SQLite 缓存：哪些文档已有向量？
  3. 只对【未命中】的文档调用 embedding API
     - 批量请求：一次 API 调用处理多篇，节省 HTTP 开销
  4. 把新计算的向量写回 SQLite
  5. 用余弦相似度给所有文档打分，重新排序
  6. 把向量语义分和 BM25 分加权合并成最终分

随着服务运行，命中率会越来越高，API 调用越来越少
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import asyncio
import logging
import math
from typing import Optional

import numpy as np

from app.core.config import settings
from app.services.embedding import embedding_service
from app.services.vector_cache import vector_cache
from app.models.search import SearchResultItem, ScoreBreakdown

logger = logging.getLogger(__name__)


class LazyReranker:
    """
    懒计算向量重排序器

    使用示例：
        reranker = LazyReranker()
        results = await reranker.rerank(
            query="人工智能医疗",
            hits=bm25_hits,       # ES 原始 hits
            index="news",
        )
    """

    async def rerank(
        self,
        query: str,
        hits: list[dict],          # ES 原始 hit 列表
        index: str,
        debug: bool = False,
    ) -> list[dict]:
        """
        主入口：对 BM25 召回结果做向量重排序

        步骤：
          1. 获取查询向量（缓存优先）
          2. 批量获取文档向量（缓存优先，缺失的批量调 API）
          3. 余弦相似度打分
          4. 和 BM25 分加权合并
          5. 按最终分降序排列
        """
        if not hits:
            return hits

        # 步骤 1：查询向量（大概率缓存命中，同一个词会被反复搜索）
        query_vector = await self._get_query_vector(query)

        # 步骤 2：文档向量（批量，核心逻辑）
        doc_ids    = [h["_id"] for h in hits]
        doc_vectors = await self._get_doc_vectors_batch(hits, index)

        # 步骤 3&4：打分 + 合并
        query_np = np.array(query_vector, dtype=np.float32)
        scored = []
        for hit in hits:
            doc_id  = hit["_id"]
            doc_vec = doc_vectors.get(doc_id)

            bm25_score = float(hit.get("_score") or 0)

            if doc_vec is not None:
                # 余弦相似度 = 两向量点积 / (向量A长度 × 向量B长度)
                # 值域 -1~1，语义搜索中通常 0.7~1.0 表示相近
                doc_np  = np.array(doc_vec, dtype=np.float32)
                cos_sim = float(self._cosine_similarity(query_np, doc_np))
            else:
                # 极少数情况：API 调用失败，向量拿不到
                # 给 0 分，不让它因为向量缺失而乱排
                cos_sim = 0.0
                logger.warning(f"文档 {doc_id} 向量获取失败，向量分设为 0")

            # 加权合并
            # BM25 分值域差异大（0~20+），余弦相似度是 0~1
            # 需要归一化后再加权，否则 BM25 会压倒向量分
            # 这里用 tanh 把 BM25 压缩到 0~1 区间，再加权
            bm25_norm    = math.tanh(bm25_score / settings.BM25_NORM_FACTOR)
            final_score  = (
                settings.RERANK_BM25_WEIGHT   * bm25_norm
                + settings.RERANK_VECTOR_WEIGHT * cos_sim
            )

            hit = dict(hit)  # 浅拷贝，不修改原始数据
            hit["_original_score"] = bm25_score
            hit["_vector_score"]   = round(cos_sim, 4)
            hit["_bm25_norm"]      = round(bm25_norm, 4)
            hit["_score"]          = round(final_score, 6)
            scored.append(hit)

        # 步骤 5：按最终分降序
        scored.sort(key=lambda h: h["_score"], reverse=True)
        return scored

    # ── 私有方法 ──────────────────────────────────────────────────

    async def _get_query_vector(self, query: str) -> list[float]:
        """
        获取查询词向量，优先读缓存
        查询词往往会被重复搜索，缓存命中率通常很高
        """
        cached = await vector_cache.get_query(query)
        if cached is not None:
            logger.debug(f"查询向量缓存命中: {query[:30]}")
            return cached

        logger.debug(f"查询向量缓存未命中，调用 API: {query[:30]}")
        vector = await embedding_service.embed(query)
        await vector_cache.set_query(query, vector)
        return vector

    async def _get_doc_vectors_batch(
        self, hits: list[dict], index: str
    ) -> dict[str, Optional[list[float]]]:
        """
        批量获取文档向量，核心流程：

        ┌──────────────────────────────────────┐
        │  所有文档 doc_ids                     │
        │     ↓                                │
        │  batch_get_docs（一次 SQL IN 查询）   │
        │     ↓                                │
        │  ┌─────────────┬──────────────────┐  │
        │  │  缓存命中    │   缓存未命中      │  │
        │  │  直接使用    │   需要调 API      │  │
        │  └─────────────┴──────────────────┘  │
        │              ↓                       │
        │  _batch_embed_missing（并发请求）     │
        │              ↓                       │
        │  batch_set_docs（写回 SQLite）        │
        └──────────────────────────────────────┘
        """
        doc_ids = [h["_id"] for h in hits]

        # 一次 SQL 查全部（比循环单个快很多）
        cached_map = await vector_cache.batch_get_docs(index, doc_ids)

        # 找出哪些文档没有缓存
        missing_ids = [did for did, vec in cached_map.items() if vec is None]

        if missing_ids:
            logger.info(
                f"文档向量：{len(doc_ids) - len(missing_ids)} 命中缓存，"
                f"{len(missing_ids)} 条需要调 API"
            )

            # 只对未命中的文档调 embedding API
            # 构建 {doc_id: 文本} 的映射
            missing_hit_map = {
                h["_id"]: h for h in hits if h["_id"] in missing_ids
            }
            new_vectors = await self._batch_embed_missing(missing_hit_map)

            # 写回缓存
            if new_vectors:
                await vector_cache.batch_set_docs(index, new_vectors)

            # 合并结果
            cached_map.update(new_vectors)
        else:
            logger.debug(f"文档向量：全部 {len(doc_ids)} 条命中缓存 🎉")

        return cached_map

    async def _batch_embed_missing(
        self, id_hit_map: dict[str, dict]
    ) -> dict[str, list[float]]:
        """
        对缺失向量的文档批量调用 embedding API

        并发策略：
          - 按 EMBED_BATCH_SIZE 分组（默认 20 个一批）
          - 组内用 asyncio.gather 并发，避免打爆 API 速率限制
          - 单个失败不影响其他文档

        为什么要分批而不是全部并发？
          大多数 embedding API 有 QPS 限制（如 OpenAI 默认 3000 RPM）
          一次发几百个并发请求会触发 429 Rate Limit
        """
        results: dict[str, list[float]] = {}
        items = list(id_hit_map.items())  # [(doc_id, hit), ...]
        batch_size = settings.EMBED_BATCH_SIZE  # 默认 20

        for i in range(0, len(items), batch_size):
            batch = items[i: i + batch_size]

            # 并发处理这一批
            tasks = [
                self._embed_single_doc(doc_id, hit)
                for doc_id, hit in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for (doc_id, _), result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    # 单个文档失败，记录日志但不中断整个流程
                    logger.error(f"文档 {doc_id} embedding 失败: {result}")
                elif result is not None:
                    results[doc_id] = result

        return results

    async def _embed_single_doc(
        self, doc_id: str, hit: dict
    ) -> Optional[list[float]]:
        """
        给单个文档生成 embedding

        文本拼接策略：标题 + 正文前 N 字
          - 标题通常最能代表文章主题，权重更高
          - 正文取前 500 字足以表达主题，也节省 token 费用
          - 用 \n 分隔，让模型知道这是两个不同部分
        """
        src     = hit.get("_source", {})
        title   = src.get(settings.FIELD_TITLE, "") or ""
        content = src.get(settings.FIELD_CONTENT, "") or ""

        # 截取正文，避免超出 embedding 模型的 token 限制
        text = f"{title}\n{content[:settings.EMBED_CONTENT_CHARS]}".strip()

        if not text:
            return None

        return await embedding_service.embed(text)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        余弦相似度计算
        公式：cos(θ) = (A·B) / (|A| × |B|)

        数值稳定性处理：
          - 如果向量范数为 0（零向量），返回 0 避免除以零
          - clip(-1, 1) 防止浮点误差导致结果略超出 [-1, 1]
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))


# 全局单例
lazy_reranker = LazyReranker()
