# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

News/blog search service built on Elasticsearch with AI-powered vector reranking. Supports 4 sorting modes and uses SQLite to persistently cache embeddings so API calls decrease over time.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start the service (the module path assumes an app/ package structure)
uvicorn app.main:app --reload --port 8000

# Health check
curl http://localhost:8000/health

# Test search
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "AI policy", "sort": "hybrid"}'

# Evict old cache entries
curl -X DELETE "http://localhost:8000/api/v1/cache/evict?keep_days=90"
```

## File Structure Note

**Important discrepancy:** `main.py` imports from `app.routers`, `app.core`, and `app.services`, but these subdirectories do not currently exist. The files `config.py`, `search_engine.py`, `reranker.py`, and `vector_cache.py` are at the root level and represent a flat/partial version of the intended `app/` package structure described in README.md. The intended full structure is:

```
app/
├── main.py
├── core/config.py, es_client.py
├── models/search.py
├── services/search_engine.py, reranker.py, vector_cache.py, embedding.py
└── routers/search.py
```

## Architecture

### Search Modes (4 sorts)

| Mode | Description | Speed |
|------|-------------|-------|
| `hybrid` | BM25 × multi-signal weights (freshness + authority + quality + hotness) | Fast |
| `hybrid_vector` | BM25 recall → lazy vector reranking (cache-first) | Slow on cold start |
| `relevance` | Pure BM25 | Fastest |
| `date` | Chronological only | Fastest |

### Lazy Vector Reranking Pipeline (`sort=hybrid_vector`)

```
BM25 recall N candidates (RERANK_RECALL_SIZE=50)
  → Batch-check SQLite for cached document vectors
    ├── Cache hit → use immediately
    └── Cache miss → batch-call embedding API → write back to SQLite
  → Compute cosine similarity
  → Merge: final = 0.4 × tanh(bm25/8.0) + 0.6 × cosine_similarity
  → Re-sort, manual pagination
```

BM25 scores are normalized via `tanh(score / BM25_NORM_FACTOR)` to map them from 0–20+ range into 0–1 so they can be fairly weighted against cosine similarity.

### SQLite Cache (`vector_cache.py`)

Two tables: `doc_vectors` (key: `"{index}:{doc_id}"`) and `query_vectors` (key: SHA256 hash of query text). Vectors stored as NumPy binary blobs (~4x smaller than JSON). Uses WAL mode and aiosqlite for non-blocking I/O. Cache file auto-created at `data/vector_cache.db` on startup.

### Multi-Signal Formula (hybrid mode)

```
score = BM25 × (weight_freshness × gauss_decay(date)
              + weight_authority × source_rank
              + weight_quality × log(content_length)
              + weight_hotness × log(click_count))
```

### Service Initialization (Lifespan pattern)

`main.py` uses FastAPI's `lifespan` context manager: on startup it pings Elasticsearch and calls `vector_cache.init()` to create SQLite tables. Global singletons (`es_client`, `vector_cache`, `search_engine`, `lazy_reranker`) are initialized at module load time.

## Configuration (`.env`)

Copy `.env.example` to `.env`. Required fields:

```env
ES_HOST=http://your-es-host:9200
DEFAULT_INDEX=news,blogs
FIELD_TITLE=title
FIELD_CONTENT=content
FIELD_DATE=published_at   # Must be ES date type

# Required for hybrid_vector mode
EMBEDDING_PROVIDER=openai  # or "siliconflow"
OPENAI_API_KEY=sk-xxxx
```

Key tuning parameters (all have defaults):

```env
RERANK_RECALL_SIZE=50         # Candidates fetched before reranking
RERANK_BM25_WEIGHT=0.4
RERANK_VECTOR_WEIGHT=0.6
BM25_NORM_FACTOR=8.0          # tanh normalization divisor
EMBED_BATCH_SIZE=20           # Reduce if hitting API rate limits
EMBED_CONTENT_CHARS=500       # Text chars used per document for embedding
WEIGHT_FRESHNESS=1.0
WEIGHT_SOURCE_RANK=1.5
FRESHNESS_SCALE=30d           # Gaussian decay half-life
```

## API Endpoints

- `POST /api/v1/search` — Main search (see README for full request/response schema)
- `GET /api/v1/indices` — List ES indices
- `GET /api/v1/cache/stats` — SQLite cache statistics
- `DELETE /api/v1/cache/evict?keep_days=90` — Evict stale document vectors
- `GET /health` — ES + cache health

The `debug_scores=true` request parameter returns per-document score breakdown (`bm25_score`, `vector_score`, `source_rank`, `final_score`).
