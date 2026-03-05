# News Search API

基于 Elasticsearch 的新闻 / 博客搜索服务，支持多信号排序和懒计算向量重排序。

## 架构概览

```
POST /api/v1/search
        │
        ▼
┌───────────────────┐
│   sort=hybrid     │  BM25 × (时效 + 权威 + 质量 + 热度)
│                   │  → 直接返回
└───────────────────┘

┌─────────────────────────────────────────────────────┐
│   sort=hybrid_vector                                │
│                                                     │
│  BM25 召回 N 篇候选                                  │
│        │                                            │
│        ▼                                            │
│  批量查 SQLite 缓存                                  │
│   ├── 命中 → 直接读向量                              │
│   └── 未命中 → 批量调 Embedding API → 写回缓存       │
│        │                                            │
│  余弦相似度打分 × BM25归一化分 → 加权合并 → 重排序    │
└─────────────────────────────────────────────────────┘
```

---

## 快速启动

**1. 安装依赖**

```bash
pip install -r requirements.txt
```

**2. 配置环境变量**

```bash
cp .env.example .env
```

打开 `.env`，至少修改以下几项：

```env
ES_HOST=http://your-es-host:9200
DEFAULT_INDEX=news,blogs        # 你的 ES 索引名
FIELD_TITLE=title               # 你的标题字段名
FIELD_CONTENT=content           # 你的正文字段名
FIELD_DATE=published_at         # 你的时间字段名（必须是 date 类型）

# 使用向量重排序时必填
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-xxxx
```

**3. 启动服务**

```bash
uvicorn app.main:app --reload --port 8000
```

启动时会自动初始化 SQLite 缓存文件（`data/vector_cache.db`）。

- 交互式 API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

---

## 排序模式

| 模式 | 说明 | 响应速度 | 适用场景 |
|------|------|----------|----------|
| `hybrid` | BM25 × 多信号加权（时效+权威+质量+热度） | 快 | 日常搜索，默认推荐 |
| `hybrid_vector` | BM25 召回 + 向量重排序，缓存优先 | 较慢（冷启动），热点快 | 自然语言提问，语义搜索 |
| `relevance` | 纯 BM25 相关度 | 最快 | 关键词精确匹配 |
| `date` | 纯时间倒序 | 最快 | 只要最新内容 |

---

## API 接口

### `POST /api/v1/search`

**请求参数**

| 参数 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `query` | string | ✅ | — | 搜索关键词或自然语言问题 |
| `indices` | string | ❌ | 全部 | 指定 ES 索引，逗号分隔，如 `"news,blogs"` |
| `date_from` | datetime | ❌ | — | 起始时间，ISO 8601 格式 |
| `date_to` | datetime | ❌ | — | 截止时间，ISO 8601 格式 |
| `sort` | string | ❌ | `hybrid` | 排序模式，见上表 |
| `page` | int | ❌ | `1` | 页码 |
| `page_size` | int | ❌ | `10` | 每页条数，最多 50 |
| `include_highlights` | bool | ❌ | `true` | 是否返回关键词高亮片段 |
| `debug_scores` | bool | ❌ | `false` | 是否返回各维度得分明细 |

**请求示例**

```bash
# 日常搜索（hybrid，最常用）
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "人工智能医疗",
    "sort": "hybrid",
    "date_from": "2024-01-01T00:00:00",
    "page_size": 10
  }'

# 语义搜索（hybrid_vector，首次调用慢，之后命中缓存变快）
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "最近有哪些关于AI监管政策的讨论",
    "sort": "hybrid_vector",
    "indices": "news",
    "debug_scores": true
  }'
```

**响应示例**

```json
{
  "query": "人工智能医疗",
  "sort_mode": "hybrid_vector",
  "total": 1280,
  "page": 1,
  "page_size": 10,
  "total_pages": 128,
  "took_ms": 340,
  "results": [
    {
      "id": "abc123",
      "title": "大模型在医疗影像诊断中的应用",
      "url": "https://example.com/article/123",
      "source": "科技日报",
      "published_at": "2024-11-20T08:00:00",
      "summary": "近年来，人工智能大模型在医疗诊断领域取得了...",
      "highlights": {
        "title": ["大模型在<em>医疗</em>影像诊断中的应用"],
        "content": ["<em>人工智能</em>在影像识别方面准确率已超过..."]
      },
      "score": 0.847291,
      "score_breakdown": {
        "final_score": 0.847291,
        "bm25_score": 12.34,
        "vector_score": 0.91,
        "source_rank": 0.9
      }
    }
  ]
}
```

### `GET /api/v1/indices`

列出所有可用 ES 索引及文档数量。

### `GET /api/v1/cache/stats`

查看 SQLite 向量缓存状态：

```json
{
  "doc_vectors_cached": 3821,
  "query_vectors_cached": 156,
  "query_total_hits": 4209,
  "db_size_mb": 18.6,
  "db_path": "data/vector_cache.db"
}
```

### `DELETE /api/v1/cache/evict?keep_days=90`

清理超过 N 天未被召回的文档向量，防止缓存无限增长。

---

## hybrid_vector 排序原理

```
最终得分 = BM25权重 × tanh(BM25分 / 归一化系数)
         + 向量权重 × 余弦相似度

默认：0.4 × tanh(bm25/8) + 0.6 × cosine_similarity
```

**为什么要归一化 BM25 分？**
BM25 原始分值域是 0~20+，余弦相似度是 0~1，直接相加会被 BM25 压倒。
`tanh` 把 BM25 压缩到 0~1 区间，两者才能公平加权。

**缓存命中率随时间增长：**

```
第 1 天：大部分文档未缓存，每次搜索都要调 API（慢）
第 7 天：热门文档已缓存，命中率上升
第 30 天：高频召回文档基本全部命中，API 调用量趋近于 0
```

---

## Level 1 多信号排序原理

```
最终得分 = BM25得分 × (时效分 + 权威分 + 质量分 + 热度分)
```

| 信号 | 计算方式 | 对应 ES 字段 | 没有该字段时 |
|------|----------|------------|------------|
| 时效分 | 高斯衰减，30天前降到 0.5 | `published_at`（date 类型） | 排序异常，必须有 |
| 权威分 | 直接读字段值（0~1） | `source_rank`（float） | 默认 0.5，所有来源相同 |
| 质量分 | `log(content_length)` | `content_length`（integer） | 不参与排序 |
| 热度分 | `log(click_count)` | `click_count`（integer） | 不参与排序 |

如果你的 ES 文档还没有 `source_rank` / `content_length` 字段，可以用迁移脚本批量补充：

```bash
# 先在 ES mapping 中添加字段（见脚本文件顶部注释）
# 再运行：
python scripts/migrate_add_vector_field.py --index news --batch 20
```

---

## 权重调参指南

修改 `.env` 中的权重，不需要改代码：

```env
# 如果结果太偏旧文章 → 增大 WEIGHT_FRESHNESS
WEIGHT_FRESHNESS=1.5

# 如果想更突出权威来源 → 增大 WEIGHT_SOURCE_RANK
WEIGHT_SOURCE_RANK=2.0

# 如果语义搜索结果和关键词不够相关 → 调大 BM25 权重
RERANK_BM25_WEIGHT=0.6
RERANK_VECTOR_WEIGHT=0.4

# 如果 API 触发速率限制 → 减小批次大小
EMBED_BATCH_SIZE=10

# 如果时效衰减太快（新闻更新很频繁）→ 缩短衰减尺度
FRESHNESS_SCALE=7d
```

---

## 项目结构

```
search-api/
├── app/
│   ├── main.py                      # FastAPI 入口，管理生命周期
│   ├── core/
│   │   ├── config.py                # 所有配置项（从 .env 读取）
│   │   └── es_client.py             # ES 异步连接单例
│   ├── models/
│   │   └── search.py                # 请求 / 响应数据模型
│   ├── services/
│   │   ├── search_engine.py         # 搜索主逻辑，整合各排序模式
│   │   ├── reranker.py              # 懒计算向量重排序（缓存优先）
│   │   ├── vector_cache.py          # SQLite 持久化向量缓存
│   │   └── embedding.py             # Embedding API 调用（OpenAI / 硅基）
│   └── routers/
│       └── search.py                # HTTP 路由层
├── scripts/
│   └── migrate_add_vector_field.py  # 历史数据批量补充向量和质量字段
├── data/                            # SQLite 缓存文件目录（自动创建）
├── tests/
│   └── test_search.py               # 测试用例
├── .env.example                     # 配置模板
├── requirements.txt
└── README.md
```

---

## 依赖说明

| 包 | 用途 |
|----|------|
| `fastapi` | Web 框架 |
| `elasticsearch[async]` | ES 异步客户端 |
| `aiosqlite` | 异步 SQLite，不阻塞事件循环 |
| `numpy` | 向量运算（余弦相似度） |
| `httpx` | 异步 HTTP 客户端，调用 Embedding API |
| `pydantic-settings` | 从 `.env` 读取配置 |
