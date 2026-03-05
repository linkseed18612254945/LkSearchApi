"""
FastAPI entry point
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import search
from app.core.es_client import es_client
from app.services.vector_cache import vector_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if await es_client.ping():
        print("Elasticsearch connected")
    else:
        print("Elasticsearch connection FAILED — check config")

    await vector_cache.init()
    stats = await vector_cache.stats()
    print(
        f"Vector cache ready: {stats['doc_vectors_cached']} doc vectors, "
        f"{stats['query_vectors_cached']} query vectors, "
        f"{stats['db_size_mb']} MB"
    )

    yield

    # Shutdown
    await es_client.close()
    await vector_cache.close()
    print("Connections closed")


app = FastAPI(
    title="News Search API",
    description="Elasticsearch + lazy vector reranking + SQLite persistent cache",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router, prefix="/api/v1", tags=["Search"])


@app.get("/health")
async def health_check():
    es_ok = await es_client.ping()
    cache_stats = await vector_cache.stats()
    return {
        "status": "ok" if es_ok else "degraded",
        "elasticsearch": "connected" if es_ok else "disconnected",
        "vector_cache": cache_stats,
    }
