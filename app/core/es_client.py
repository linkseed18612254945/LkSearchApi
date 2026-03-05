"""
Elasticsearch async client wrapper
"""

import logging
from typing import Optional

from elasticsearch import AsyncElasticsearch

from app.core.config import settings

logger = logging.getLogger(__name__)


class ESClient:
    def __init__(self):
        self._client: Optional[AsyncElasticsearch] = None

    def _get(self) -> AsyncElasticsearch:
        if self._client is None:
            kwargs: dict = {"hosts": [settings.ES_HOST]}
            if settings.ES_USERNAME:
                kwargs["basic_auth"] = (settings.ES_USERNAME, settings.ES_PASSWORD)
            self._client = AsyncElasticsearch(**kwargs)
        return self._client

    async def ping(self) -> bool:
        try:
            return await self._get().ping()
        except Exception as e:
            logger.error(f"ES ping failed: {e}")
            return False

    async def search(self, index: str, body: dict, size: int = 10, from_: int = 0) -> dict:
        """
        Execute a search. body keys (query, sort, highlight, etc.) are
        forwarded directly as keyword arguments to the ES client.
        """
        return await self._get().search(
            index=index,
            size=size,
            from_=from_,
            **body,
        )

    async def cat_indices(self, index: str = "*") -> list:
        resp = await self._get().cat.indices(index=index, format="json", h="index,docs.count,store.size,health")
        return resp.body if hasattr(resp, "body") else list(resp)

    async def close(self):
        if self._client:
            await self._client.close()
            self._client = None


es_client = ESClient()
