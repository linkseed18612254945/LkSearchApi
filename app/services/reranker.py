"""
Lazy vector reranker:
  BM25 recall -> batch cache check -> API for misses -> cosine similarity -> weighted merge
"""

import asyncio
import logging
import math
from typing import Optional

import numpy as np

from app.core.config import settings
from app.services.embedding import embedding_service
from app.services.vector_cache import vector_cache

logger = logging.getLogger(__name__)


class LazyReranker:

    async def rerank(
        self,
        query: str,
        hits: list[dict],
        index: str,
        debug: bool = False,
    ) -> list[dict]:
        if not hits:
            return hits

        query_vector = await self._get_query_vector(query)
        doc_vectors = await self._get_doc_vectors_batch(hits, index)

        query_np = np.array(query_vector, dtype=np.float32)
        scored = []
        for hit in hits:
            doc_id = hit["_id"]
            doc_vec = doc_vectors.get(doc_id)
            bm25_score = float(hit.get("_score") or 0)

            if doc_vec is not None:
                doc_np = np.array(doc_vec, dtype=np.float32)
                cos_sim = float(self._cosine_similarity(query_np, doc_np))
            else:
                cos_sim = 0.0
                logger.warning(f"No vector for doc {doc_id}, vector_score=0")

            bm25_norm = math.tanh(bm25_score / settings.BM25_NORM_FACTOR)
            final_score = (
                settings.RERANK_BM25_WEIGHT * bm25_norm
                + settings.RERANK_VECTOR_WEIGHT * cos_sim
            )

            hit = dict(hit)
            hit["_original_score"] = bm25_score
            hit["_vector_score"] = round(cos_sim, 4)
            hit["_bm25_norm"] = round(bm25_norm, 4)
            hit["_score"] = round(final_score, 6)
            scored.append(hit)

        scored.sort(key=lambda h: h["_score"], reverse=True)
        return scored

    async def _get_query_vector(self, query: str) -> list[float]:
        cached = await vector_cache.get_query(query)
        if cached is not None:
            return cached
        vector = await embedding_service.embed(query)
        await vector_cache.set_query(query, vector)
        return vector

    async def _get_doc_vectors_batch(
        self, hits: list[dict], index: str
    ) -> dict[str, Optional[list[float]]]:
        doc_ids = [h["_id"] for h in hits]
        cached_map = await vector_cache.batch_get_docs(index, doc_ids)
        missing_ids = [did for did, vec in cached_map.items() if vec is None]

        if missing_ids:
            logger.info(
                f"Doc vectors: {len(doc_ids) - len(missing_ids)} cached, "
                f"{len(missing_ids)} need API calls"
            )
            missing_hit_map = {h["_id"]: h for h in hits if h["_id"] in missing_ids}
            new_vectors = await self._batch_embed_missing(missing_hit_map)
            if new_vectors:
                await vector_cache.batch_set_docs(index, new_vectors)
            cached_map.update(new_vectors)

        return cached_map

    async def _batch_embed_missing(
        self, id_hit_map: dict[str, dict]
    ) -> dict[str, list[float]]:
        results: dict[str, list[float]] = {}
        items = list(id_hit_map.items())
        batch_size = settings.EMBED_BATCH_SIZE

        for i in range(0, len(items), batch_size):
            batch = items[i: i + batch_size]
            tasks = [self._embed_single_doc(doc_id, hit) for doc_id, hit in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for (doc_id, _), result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Embedding failed for doc {doc_id}: {result}")
                elif result is not None:
                    results[doc_id] = result

        return results

    async def _embed_single_doc(self, doc_id: str, hit: dict) -> Optional[list[float]]:
        src = hit.get("_source", {})
        title = src.get(settings.FIELD_TITLE, "") or ""
        content = src.get(settings.FIELD_CONTENT, "") or ""
        text = f"{title}\n{content[:settings.EMBED_CONTENT_CHARS]}".strip()
        if not text:
            return None
        return await embedding_service.embed(text)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))


lazy_reranker = LazyReranker()
