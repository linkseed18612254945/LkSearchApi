"""
配置管理 — 所有参数从 .env 文件读取
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Elasticsearch ──────────────────────────────────────────
    ES_HOST: str = "http://localhost:9200"
    ES_USERNAME: str = ""
    ES_PASSWORD: str = ""

    # ── 搜索默认参数 ────────────────────────────────────────────
    DEFAULT_INDEX: str = "news,blogs"
    DEFAULT_PAGE_SIZE: int = 10
    MAX_PAGE_SIZE: int = 50

    # ── ES 字段名映射（改成你实际的字段名）──────────────────────
    FIELD_TITLE: str = "title"
    FIELD_CONTENT: str = "content"
    FIELD_DATE: str = "published_at"
    FIELD_SOURCE: str = "source"
    FIELD_URL: str = "url"
    FIELD_SOURCE_RANK: str = "source_rank"
    FIELD_CONTENT_LENGTH: str = "content_length"
    FIELD_CLICK_COUNT: str = "click_count"

    # ── Level 1：多信号加权 ─────────────────────────────────────
    WEIGHT_FRESHNESS: float = 1.0
    WEIGHT_SOURCE_RANK: float = 1.5
    WEIGHT_QUALITY: float = 0.8
    WEIGHT_HOTNESS: float = 0.5
    FRESHNESS_DECAY: float = 0.5
    FRESHNESS_SCALE: str = "30d"

    # ── Embedding API ───────────────────────────────────────────
    EMBEDDING_PROVIDER: str = "openai"       # "openai" 或 "siliconflow"
    OPENAI_API_KEY: str = ""
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    SILICONFLOW_API_KEY: str = ""
    SILICONFLOW_EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"
    SILICONFLOW_BASE_URL: str = "https://api.siliconflow.cn/v1"

    # ── 懒计算重排序配置 ─────────────────────────────────────────
    # BM25 召回多少篇文章参与重排序（比最终返回数量多，给重排序足够候选）
    # 例如：用户要 10 条结果，召回 50 篇重排后取前 10
    RERANK_RECALL_SIZE: int = 50

    # 重排序得分加权（两个权重之和建议为 1.0）
    # BM25 分先用 tanh 归一化到 0~1，再加权
    RERANK_BM25_WEIGHT: float = 0.4     # BM25 关键词相关度权重
    RERANK_VECTOR_WEIGHT: float = 0.6   # 向量语义相似度权重（语义更重要）

    # tanh 归一化系数：BM25 原始分 / 这个数，再取 tanh
    # BM25 分通常在 0~15 之间，除以 8 后大多数值落在 tanh 的敏感区间
    BM25_NORM_FACTOR: float = 8.0

    # embedding API 并发批次大小（每批多少个文档并发请求）
    # 太大会触发 API 速率限制，太小速度慢，20 是比较安全的值
    EMBED_BATCH_SIZE: int = 20

    # 生成向量时截取正文的字符数（节省 token，500 字通常足以表达主题）
    EMBED_CONTENT_CHARS: int = 500

    # ── SQLite 向量缓存 ──────────────────────────────────────────
    # 缓存文件路径（相对于项目根目录，会自动创建）
    VECTOR_CACHE_PATH: str = "data/vector_cache.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
