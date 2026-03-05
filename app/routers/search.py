"""
Search router — all /api/v1/* endpoints
"""

from fastapi import APIRouter, Body, HTTPException, Query

from app.core.es_client import es_client
from app.services.search_engine import search_engine
from app.services.vector_cache import vector_cache
from app.models.search import SearchRequest, SearchResponse

router = APIRouter()

_SEARCH_EXAMPLES = {
    "1_hybrid_basic": {
        "summary": "Hybrid — Ukraine war (default mode)",
        "description": "Multi-signal BM25: freshness + hotness boost. Fast.",
        "value": {
            "query": "Ukraine war ceasefire negotiations",
            "sort": "hybrid",
            "page": 1,
            "page_size": 5,
        },
    },
    "2_relevance": {
        "summary": "Relevance — AI policy (pure BM25)",
        "description": "Pure keyword relevance, fastest mode.",
        "value": {
            "query": "artificial intelligence regulation policy",
            "sort": "relevance",
            "page": 1,
            "page_size": 5,
        },
    },
    "3_date": {
        "summary": "Date — latest North Korea news",
        "description": "Chronological (newest first), BM25 as tiebreaker.",
        "value": {
            "query": "North Korea missile nuclear",
            "sort": "date",
            "page": 1,
            "page_size": 5,
        },
    },
    "4_date_filter": {
        "summary": "Hybrid — climate change (date range filter)",
        "description": "Hybrid mode restricted to articles from 2025 onward.",
        "value": {
            "query": "climate change global warming",
            "sort": "hybrid",
            "page": 1,
            "page_size": 5,
            "date_from": "2025-01-01",
        },
    },
    "5_debug_scores": {
        "summary": "Hybrid — debug score breakdown",
        "description": "Returns per-document bm25_score, vector_score, source_rank.",
        "value": {
            "query": "China economy trade war tariffs",
            "sort": "hybrid",
            "page": 1,
            "page_size": 5,
            "debug_scores": True,
        },
    },
    "6_pagination": {
        "summary": "Relevance — page 2 results",
        "description": "Demonstrates pagination with page/page_size.",
        "value": {
            "query": "cryptocurrency bitcoin ethereum",
            "sort": "relevance",
            "page": 2,
            "page_size": 5,
        },
    },
    "7_highlights": {
        "summary": "Relevance — with highlights disabled",
        "description": "Highlights off — faster response, smaller payload.",
        "value": {
            "query": "Middle East conflict Gaza",
            "sort": "relevance",
            "page": 1,
            "page_size": 5,
            "include_highlights": False,
        },
    },
    "8_chinese_query": {
        "summary": "Hybrid — Chinese query (中文搜索)",
        "description": "t_news contains bilingual content; Chinese queries work natively.",
        "value": {
            "query": "人工智能 政策 监管",
            "sort": "hybrid",
            "page": 1,
            "page_size": 5,
        },
    },
    "9_hybrid_vector_en": {
        "summary": "Hybrid Vector — semantic rerank (English)",
        "description": "BM25 recalls 50 candidates, SiliconFlow bge-large-zh-v1.5 reranks by cosine similarity. Slower on cold start, cached on repeat.",
        "value": {
            "query": "North Korea Russia military cooperation",
            "sort": "hybrid_vector",
            "page": 1,
            "page_size": 5,
            "debug_scores": True,
        },
    },
    "10_hybrid_vector_zh": {
        "summary": "Hybrid Vector — semantic rerank (中文)",
        "description": "语义向量重排序，适合中文长尾查询，首次慢（调 API），之后走缓存。",
        "value": {
            "query": "人工智能 医疗 应用",
            "sort": "hybrid_vector",
            "page": 1,
            "page_size": 5,
            "debug_scores": True,
        },
    },
}


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest = Body(..., openapi_examples=_SEARCH_EXAMPLES)):
    try:
        return await search_engine.search(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indices")
async def list_indices():
    try:
        indices = await es_client.cat_indices()
        return {"indices": indices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats")
async def cache_stats():
    return await vector_cache.stats()


@router.delete("/cache/evict")
async def evict_cache(keep_days: int = Query(90, ge=1, description="Keep vectors newer than this many days")):
    deleted = await vector_cache.evict_old_docs(keep_days=keep_days)
    return {"deleted": deleted, "keep_days": keep_days}
