"""
向量持久化缓存（SQLite）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
为什么用 SQLite 而不是内存字典？
  内存字典：服务重启就全丢，等于每次冷启动
  SQLite：写到本地文件，永久保留，零额外依赖

两张表：
  doc_vectors   — key="{index}:{doc_id}"，存文档内容的向量
  query_vectors — key=SHA256(查询词)，存搜索词的向量

向量以 numpy 二进制格式（.npy bytes）存储，比 JSON 小 ~4x，读写更快

并发安全：
  aiosqlite 是异步的，不会阻塞 FastAPI 的事件循环
  写操作用 WAL 模式（Write-Ahead Logging），允许多读单写并发
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import hashlib
import io
import time
import logging
from pathlib import Path
from typing import Optional

import aiosqlite
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorCache:
    """
    SQLite 向量缓存
    使用方式：
        cache = VectorCache()
        await cache.init()   # 服务启动时调用一次
        vec = await cache.get_doc("news", "doc_abc123")
        await cache.set_doc("news", "doc_abc123", [0.1, 0.2, ...])
    """

    def __init__(self):
        self.db_path = Path(settings.VECTOR_CACHE_PATH)
        self._db: Optional[aiosqlite.Connection] = None

    # ── 生命周期 ──────────────────────────────────────────────────

    async def init(self):
        """
        初始化：打开数据库连接，建表（如果不存在）
        在 FastAPI lifespan 启动时调用
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))

        # WAL 模式：写时不锁读，适合并发搜索场景
        await self._db.execute("PRAGMA journal_mode=WAL")
        # 内存缓存 64MB，加快频繁读取
        await self._db.execute("PRAGMA cache_size=-65536")

        await self._db.executescript("""
            -- 文档向量表：存文章内容的向量
            CREATE TABLE IF NOT EXISTS doc_vectors (
                cache_key  TEXT PRIMARY KEY,   -- "{index}:{doc_id}"
                vector     BLOB NOT NULL,       -- numpy 二进制
                created_at REAL NOT NULL        -- unix timestamp，用于统计
            );

            -- 查询向量表：存用户搜索词的向量
            CREATE TABLE IF NOT EXISTS query_vectors (
                cache_key  TEXT PRIMARY KEY,   -- SHA256(查询词)
                query_text TEXT NOT NULL,       -- 原始查询词（方便调试）
                vector     BLOB NOT NULL,
                hit_count  INTEGER DEFAULT 1,   -- 被命中多少次（了解热门查询）
                created_at REAL NOT NULL,
                last_hit   REAL NOT NULL
            );

            -- 索引：按创建时间查询（用于清理旧缓存）
            CREATE INDEX IF NOT EXISTS idx_doc_created   ON doc_vectors(created_at);
            CREATE INDEX IF NOT EXISTS idx_query_created ON query_vectors(created_at);
        """)
        await self._db.commit()
        logger.info(f"✅ 向量缓存已初始化：{self.db_path}")

    async def close(self):
        if self._db:
            await self._db.close()
            self._db = None

    # ── 文档向量：get / set / batch_get ──────────────────────────

    async def get_doc(self, index: str, doc_id: str) -> Optional[list[float]]:
        """读取单个文档向量，未命中返回 None"""
        key = self._doc_key(index, doc_id)
        return await self._get(key, table="doc_vectors")

    async def set_doc(self, index: str, doc_id: str, vector: list[float]):
        """写入单个文档向量"""
        key = self._doc_key(index, doc_id)
        await self._set_doc_internal(key, vector)

    async def batch_get_docs(
        self, index: str, doc_ids: list[str]
    ) -> dict[str, Optional[list[float]]]:
        """
        批量读取文档向量
        返回 {doc_id: vector_or_None}
        比循环单个 get 快很多（一次 SQL IN 查询）
        """
        if not doc_ids:
            return {}

        keys   = [self._doc_key(index, did) for did in doc_ids]
        id_map = {self._doc_key(index, did): did for did in doc_ids}

        placeholders = ",".join("?" * len(keys))
        async with self._db.execute(
            f"SELECT cache_key, vector FROM doc_vectors WHERE cache_key IN ({placeholders})",
            keys,
        ) as cursor:
            rows = await cursor.fetchall()

        result: dict[str, Optional[list[float]]] = {did: None for did in doc_ids}
        for cache_key, blob in rows:
            doc_id = id_map[cache_key]
            result[doc_id] = self._blob_to_vector(blob)

        hit = sum(1 for v in result.values() if v is not None)
        logger.debug(f"batch_get_docs: {hit}/{len(doc_ids)} 命中缓存")
        return result

    async def batch_set_docs(self, index: str, id_vector_map: dict[str, list[float]]):
        """
        批量写入文档向量
        用 INSERT OR REPLACE 处理已存在的 key（幂等）
        """
        if not id_vector_map:
            return

        now = time.time()
        rows = [
            (self._doc_key(index, doc_id), self._vector_to_blob(vec), now)
            for doc_id, vec in id_vector_map.items()
        ]
        await self._db.executemany(
            "INSERT OR REPLACE INTO doc_vectors (cache_key, vector, created_at) VALUES (?,?,?)",
            rows,
        )
        await self._db.commit()
        logger.debug(f"batch_set_docs: 写入 {len(rows)} 条")

    # ── 查询向量：get / set ───────────────────────────────────────

    async def get_query(self, query_text: str) -> Optional[list[float]]:
        """读取查询词向量，同时更新命中计数"""
        key = self._query_key(query_text)
        async with self._db.execute(
            "SELECT vector FROM query_vectors WHERE cache_key = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        # 更新命中统计（异步，不影响主流程）
        now = time.time()
        await self._db.execute(
            "UPDATE query_vectors SET hit_count=hit_count+1, last_hit=? WHERE cache_key=?",
            (now, key),
        )
        await self._db.commit()
        return self._blob_to_vector(row[0])

    async def set_query(self, query_text: str, vector: list[float]):
        """写入查询词向量"""
        key = self._query_key(query_text)
        now = time.time()
        await self._db.execute(
            """INSERT OR REPLACE INTO query_vectors
               (cache_key, query_text, vector, hit_count, created_at, last_hit)
               VALUES (?,?,?,1,?,?)""",
            (key, query_text[:500], self._vector_to_blob(vector), now, now),
        )
        await self._db.commit()

    # ── 缓存管理 ─────────────────────────────────────────────────

    async def stats(self) -> dict:
        """返回缓存统计信息"""
        async with self._db.execute("SELECT COUNT(*) FROM doc_vectors") as c:
            doc_count = (await c.fetchone())[0]
        async with self._db.execute("SELECT COUNT(*), SUM(hit_count) FROM query_vectors") as c:
            row = await c.fetchone()
            query_count, total_hits = row[0], (row[1] or 0)

        db_size_mb = round(self.db_path.stat().st_size / 1024 / 1024, 2) if self.db_path.exists() else 0

        return {
            "doc_vectors_cached": doc_count,
            "query_vectors_cached": query_count,
            "query_total_hits": total_hits,
            "db_size_mb": db_size_mb,
            "db_path": str(self.db_path),
        }

    async def evict_old_docs(self, keep_days: int = 90):
        """
        清理超过 keep_days 天未被召回的文档向量
        用于定期维护，避免数据库无限增长
        """
        cutoff = time.time() - keep_days * 86400
        async with self._db.execute(
            "DELETE FROM doc_vectors WHERE created_at < ?", (cutoff,)
        ) as cursor:
            deleted = cursor.rowcount
        await self._db.commit()
        logger.info(f"evict_old_docs: 清理 {deleted} 条超过 {keep_days} 天的缓存")
        return deleted

    # ── 内部工具 ─────────────────────────────────────────────────

    def _doc_key(self, index: str, doc_id: str) -> str:
        return f"{index}:{doc_id}"

    def _query_key(self, query_text: str) -> str:
        # 用 SHA256 作为 key，避免特殊字符和超长 key 问题
        return hashlib.sha256(query_text.strip().lower().encode()).hexdigest()

    def _vector_to_blob(self, vector: list[float]) -> bytes:
        """list[float] → numpy float32 二进制，比 JSON 小 ~4x"""
        buf = io.BytesIO()
        np.save(buf, np.array(vector, dtype=np.float32))
        return buf.getvalue()

    def _blob_to_vector(self, blob: bytes) -> list[float]:
        """numpy 二进制 → list[float]"""
        buf = io.BytesIO(blob)
        return np.load(buf).tolist()

    async def _get(self, key: str, table: str) -> Optional[list[float]]:
        async with self._db.execute(
            f"SELECT vector FROM {table} WHERE cache_key = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
        return self._blob_to_vector(row[0]) if row else None

    async def _set_doc_internal(self, key: str, vector: list[float]):
        now = time.time()
        await self._db.execute(
            "INSERT OR REPLACE INTO doc_vectors (cache_key, vector, created_at) VALUES (?,?,?)",
            (key, self._vector_to_blob(vector), now),
        )
        await self._db.commit()


# 全局单例
vector_cache = VectorCache()
