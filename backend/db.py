"""
db.py — asyncpg connection pool + query helpers
"""
import os
import asyncpg
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/forest_monitor"
)

pool: asyncpg.Pool | None = None


async def init_db():
    global pool
    pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )
    logger.info("Database pool initialised")


async def close_db():
    global pool
    if pool:
        await pool.close()
        logger.info("Database pool closed")


@asynccontextmanager
async def get_conn():
    """Yield a connection from the pool."""
    async with pool.acquire() as conn:
        yield conn


# ------------------------------------------------------------------ #
#  Devices                                                             #
# ------------------------------------------------------------------ #

async def get_all_devices() -> list[dict]:
    async with get_conn() as conn:
        rows = await conn.fetch(
            "SELECT * FROM devices ORDER BY device_id"
        )
        return [dict(r) for r in rows]


async def get_device(device_id: str) -> dict | None:
    async with get_conn() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM devices WHERE device_id = $1", device_id
        )
        return dict(row) if row else None


async def update_device_last_active(device_id: str):
    async with get_conn() as conn:
        await conn.execute(
            "UPDATE devices SET last_active = NOW() WHERE device_id = $1",
            device_id,
        )


async def verify_device_api_key(device_id: str, api_key: str) -> bool:
    async with get_conn() as conn:
        row = await conn.fetchrow(
            "SELECT 1 FROM devices WHERE device_id=$1 AND api_key=$2",
            device_id, api_key,
        )
        return row is not None


# ------------------------------------------------------------------ #
#  Logs                                                                #
# ------------------------------------------------------------------ #

async def insert_log(
    device_id: str,
    file_path: str,
    duration_seconds: int | None,
    file_size_bytes: int | None,
) -> int:
    async with get_conn() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO logs (device_id, file_path, duration_seconds, file_size_bytes)
            VALUES ($1, $2, $3, $4)
            RETURNING log_id
            """,
            device_id, file_path, duration_seconds, file_size_bytes,
        )
        return row["log_id"]


async def get_logs(limit: int = 50, offset: int = 0) -> list[dict]:
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT l.*, d.name AS device_name, d.latitude, d.longitude
            FROM logs l
            JOIN devices d ON l.device_id = d.device_id
            ORDER BY l.timestamp DESC
            LIMIT $1 OFFSET $2
            """,
            limit, offset,
        )
        return [dict(r) for r in rows]


async def get_log_by_id(log_id: int) -> dict | None:
    async with get_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT l.*, d.latitude, d.longitude
            FROM logs l
            JOIN devices d ON l.device_id = d.device_id
            WHERE l.log_id = $1
            """,
            log_id,
        )
        return dict(row) if row else None


async def fetch_unprocessed_logs(batch: int = 5) -> list[dict]:
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT l.log_id, l.device_id, l.file_path, l.timestamp,
                   d.latitude, d.longitude
            FROM logs l
            JOIN devices d ON l.device_id = d.device_id
            WHERE l.processed = FALSE
            ORDER BY l.timestamp ASC
            LIMIT $1
            """,
            batch,
        )
        return [dict(r) for r in rows]


async def mark_log_processed(log_id: int):
    async with get_conn() as conn:
        await conn.execute(
            "UPDATE logs SET processed = TRUE WHERE log_id = $1", log_id
        )


# ------------------------------------------------------------------ #
#  Positives & Negatives                                               #
# ------------------------------------------------------------------ #

async def insert_positive(
    log_id: int,
    device_id: str,
    timestamp,
    confidence: float,
    latitude: float,
    longitude: float,
    label: str | None = None,
) -> int:
    async with get_conn() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO positives
                (log_id, device_id, timestamp, confidence, latitude, longitude, label)
            VALUES ($1,$2,$3,$4,$5,$6,$7)
            RETURNING id
            """,
            log_id, device_id, timestamp, confidence, latitude, longitude, label,
        )
        return row["id"]


async def insert_negative(
    log_id: int, device_id: str, timestamp, confidence: float
):
    async with get_conn() as conn:
        await conn.execute(
            """
            INSERT INTO negatives (log_id, device_id, timestamp, confidence)
            VALUES ($1,$2,$3,$4)
            """,
            log_id, device_id, timestamp, confidence,
        )


async def get_positives(limit: int = 100) -> list[dict]:
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT p.*, l.file_path, d.name AS device_name
            FROM positives p
            JOIN logs l ON p.log_id = l.log_id
            JOIN devices d ON p.device_id = d.device_id
            ORDER BY p.timestamp DESC
            LIMIT $1
            """,
            limit,
        )
        return [dict(r) for r in rows]


async def get_unalerted_positives() -> list[dict]:
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT p.*, l.file_path, d.name AS device_name
            FROM positives p
            JOIN logs l ON p.log_id = l.log_id
            JOIN devices d ON p.device_id = d.device_id
            WHERE p.alerted = FALSE
            ORDER BY p.timestamp ASC
            """
        )
        return [dict(r) for r in rows]


async def mark_positives_alerted(ids: list[int]):
    async with get_conn() as conn:
        await conn.execute(
            "UPDATE positives SET alerted = TRUE WHERE id = ANY($1::int[])", ids
        )


# ------------------------------------------------------------------ #
#  Device Commands                                                     #
# ------------------------------------------------------------------ #

async def insert_command(device_id: str, command: str, payload: dict) -> int:
    async with get_conn() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO device_commands (device_id, command, payload)
            VALUES ($1, $2, $3)
            RETURNING id
            """,
            device_id, command, payload,
        )
        return row["id"]


async def get_pending_commands(device_id: str) -> list[dict]:
    async with get_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM device_commands
            WHERE device_id = $1 AND status = 'pending'
            ORDER BY created_at ASC
            """,
            device_id,
        )
        return [dict(r) for r in rows]


async def mark_command_sent(command_id: int):
    async with get_conn() as conn:
        await conn.execute(
            "UPDATE device_commands SET status='sent' WHERE id=$1", command_id
        )
