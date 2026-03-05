"""
FastAPI 主入口
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routers import search
from app.core.es_client import es_client
from app.services.vector_cache import vector_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 启动 ─────────────────────────────────────────────────
    if await es_client.ping():
        print("✅ Elasticsearch 连接成功")
    else:
        print("❌ Elasticsearch 连接失败，请检查配置")

    await vector_cache.init()   # 初始化 SQLite 缓存（建表等）
    stats = await vector_cache.stats()
    print(
        f"✅ 向量缓存已就绪：文档向量 {stats['doc_vectors_cached']} 条，"
        f"查询向量 {stats['query_vectors_cached']} 条，"
        f"数据库 {stats['db_size_mb']} MB"
    )

    yield

    # ── 关闭 ─────────────────────────────────────────────────
    await es_client.close()
    await vector_cache.close()
    print("🔌 连接已释放")


app = FastAPI(
    title="News Search API",
    description="ES + 懒计算向量重排序 + SQLite 持久化缓存",
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
    es_ok    = await es_client.ping()
    cache_stats = await vector_cache.stats()
    return {
        "status": "ok" if es_ok else "degraded",
        "elasticsearch": "connected" if es_ok else "disconnected",
        "vector_cache": cache_stats,
    }
