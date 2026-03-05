"""
Vector persistence cache (SQLite)
Two tables: doc_vectors and query_vectors.
Vectors stored as numpy binary blobs (~4x smaller than JSON).
Uses WAL mode and aiosqlite for non-blocking I/O.
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

    def __init__(self):
        self.db_path = Path(settings.VECTOR_CACHE_PATH)
        self._db: Optional[aiosqlite.Connection] = None

    async def init(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA cache_size=-65536")
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS doc_vectors (
                cache_key  TEXT PRIMARY KEY,
                vector     BLOB NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS query_vectors (
                cache_key  TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                vector     BLOB NOT NULL,
                hit_count  INTEGER DEFAULT 1,
                created_at REAL NOT NULL,
                last_hit   REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_doc_created   ON doc_vectors(created_at);
            CREATE INDEX IF NOT EXISTS idx_query_created ON query_vectors(created_at);
        """)
        await self._db.commit()
        logger.info(f"Vector cache initialized: {self.db_path}")

    async def close(self):
        if self._db:
            await self._db.close()
            self._db = None

    async def get_doc(self, index: str, doc_id: str) -> Optional[list[float]]:
        return await self._get(self._doc_key(index, doc_id), "doc_vectors")

    async def set_doc(self, index: str, doc_id: str, vector: list[float]):
        key = self._doc_key(index, doc_id)
        await self._db.execute(
            "INSERT OR REPLACE INTO doc_vectors (cache_key, vector, created_at) VALUES (?,?,?)",
            (key, self._to_blob(vector), time.time()),
        )
        await self._db.commit()

    async def batch_get_docs(self, index: str, doc_ids: list[str]) -> dict[str, Optional[list[float]]]:
        if not doc_ids:
            return {}
        keys = [self._doc_key(index, did) for did in doc_ids]
        id_map = {self._doc_key(index, did): did for did in doc_ids}
        placeholders = ",".join("?" * len(keys))
        async with self._db.execute(
            f"SELECT cache_key, vector FROM doc_vectors WHERE cache_key IN ({placeholders})",
            keys,
        ) as cursor:
            rows = await cursor.fetchall()
        result: dict[str, Optional[list[float]]] = {did: None for did in doc_ids}
        for cache_key, blob in rows:
            result[id_map[cache_key]] = self._from_blob(blob)
        return result

    async def batch_set_docs(self, index: str, id_vector_map: dict[str, list[float]]):
        if not id_vector_map:
            return
        now = time.time()
        rows = [
            (self._doc_key(index, doc_id), self._to_blob(vec), now)
            for doc_id, vec in id_vector_map.items()
        ]
        await self._db.executemany(
            "INSERT OR REPLACE INTO doc_vectors (cache_key, vector, created_at) VALUES (?,?,?)",
            rows,
        )
        await self._db.commit()

    async def get_query(self, query_text: str) -> Optional[list[float]]:
        key = self._query_key(query_text)
        async with self._db.execute(
            "SELECT vector FROM query_vectors WHERE cache_key = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        now = time.time()
        await self._db.execute(
            "UPDATE query_vectors SET hit_count=hit_count+1, last_hit=? WHERE cache_key=?",
            (now, key),
        )
        await self._db.commit()
        return self._from_blob(row[0])

    async def set_query(self, query_text: str, vector: list[float]):
        key = self._query_key(query_text)
        now = time.time()
        await self._db.execute(
            """INSERT OR REPLACE INTO query_vectors
               (cache_key, query_text, vector, hit_count, created_at, last_hit)
               VALUES (?,?,?,1,?,?)""",
            (key, query_text[:500], self._to_blob(vector), now, now),
        )
        await self._db.commit()

    async def stats(self) -> dict:
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

    async def evict_old_docs(self, keep_days: int = 90) -> int:
        cutoff = time.time() - keep_days * 86400
        async with self._db.execute(
            "DELETE FROM doc_vectors WHERE created_at < ?", (cutoff,)
        ) as cursor:
            deleted = cursor.rowcount
        await self._db.commit()
        return deleted

    def _doc_key(self, index: str, doc_id: str) -> str:
        return f"{index}:{doc_id}"

    def _query_key(self, query_text: str) -> str:
        return hashlib.sha256(query_text.strip().lower().encode()).hexdigest()

    def _to_blob(self, vector: list[float]) -> bytes:
        buf = io.BytesIO()
        np.save(buf, np.array(vector, dtype=np.float32))
        return buf.getvalue()

    def _from_blob(self, blob: bytes) -> list[float]:
        return np.load(io.BytesIO(blob)).tolist()

    async def _get(self, key: str, table: str) -> Optional[list[float]]:
        async with self._db.execute(
            f"SELECT vector FROM {table} WHERE cache_key = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
        return self._from_blob(row[0]) if row else None


vector_cache = VectorCache()
