"""
Configuration — all parameters read from .env
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Elasticsearch
    ES_HOST: str = "http://localhost:9200"
    ES_USERNAME: str = ""
    ES_PASSWORD: str = ""

    # Search defaults
    DEFAULT_INDEX: str = "t_news"
    DEFAULT_PAGE_SIZE: int = 10
    MAX_PAGE_SIZE: int = 50

    # ES field name mapping
    FIELD_TITLE: str = "c_title"
    FIELD_CONTENT: str = "c_content"
    FIELD_DATE: str = "c_publishtime_bj"
    FIELD_SOURCE: str = "c_sitename"
    FIELD_URL: str = "c_url"
    FIELD_SOURCE_RANK: str = "source_rank"
    FIELD_CONTENT_LENGTH: str = "content_length"
    FIELD_CLICK_COUNT: str = "c_readnum"

    # Multi-signal weights (hybrid mode)
    WEIGHT_FRESHNESS: float = 1.0
    WEIGHT_SOURCE_RANK: float = 0.5
    WEIGHT_QUALITY: float = 0.3
    WEIGHT_HOTNESS: float = 0.5
    FRESHNESS_DECAY: float = 0.5
    FRESHNESS_SCALE: str = "30d"

    # Embedding API
    EMBEDDING_PROVIDER: str = "openai"
    OPENAI_API_KEY: str = ""
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    SILICONFLOW_API_KEY: str = ""
    SILICONFLOW_EMBEDDING_MODEL: str = "BAAI/bge-large-zh-v1.5"
    SILICONFLOW_BASE_URL: str = "https://api.siliconflow.cn/v1"

    # Lazy rerank config
    RERANK_RECALL_SIZE: int = 50
    RERANK_BM25_WEIGHT: float = 0.4
    RERANK_VECTOR_WEIGHT: float = 0.6
    BM25_NORM_FACTOR: float = 8.0
    EMBED_BATCH_SIZE: int = 20
    EMBED_CONTENT_CHARS: int = 500

    # SQLite vector cache
    VECTOR_CACHE_PATH: str = "data/vector_cache.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
