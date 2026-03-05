"""
Embedding service — supports OpenAI and SiliconFlow providers
"""

import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:

    async def embed(self, text: str) -> list[float]:
        provider = settings.EMBEDDING_PROVIDER.lower()
        if provider == "openai":
            return await self._embed_openai(text)
        elif provider == "siliconflow":
            return await self._embed_siliconflow(text)
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    async def _embed_openai(self, text: str) -> list[float]:
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
                json={"model": settings.OPENAI_EMBEDDING_MODEL, "input": text},
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]

    async def _embed_siliconflow(self, text: str) -> list[float]:
        if not settings.SILICONFLOW_API_KEY:
            raise RuntimeError("SILICONFLOW_API_KEY is not configured")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{settings.SILICONFLOW_BASE_URL}/embeddings",
                headers={"Authorization": f"Bearer {settings.SILICONFLOW_API_KEY}"},
                json={"model": settings.SILICONFLOW_EMBEDDING_MODEL, "input": text},
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]


embedding_service = EmbeddingService()
