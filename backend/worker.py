"""
worker.py — Background inference worker
Run this as a separate process: python worker.py

Polls the logs table for unprocessed entries, runs model inference,
and writes results to positives / negatives tables.
"""
import asyncio
import logging
import os
import signal
import sys

import db
from model2 import get_classifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("worker")

WORKER_INTERVAL = int(os.getenv("WORKER_INTERVAL_SEC", "5"))
BATCH_SIZE      = int(os.getenv("WORKER_BATCH_SIZE", "5"))

_running = True


def _shutdown(signum, frame):
    global _running
    logger.info("Shutdown signal received — finishing current batch …")
    _running = False


signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


# ------------------------------------------------------------------ #

async def process_batch():
    logs = await db.fetch_unprocessed_logs(BATCH_SIZE)
    if not logs:
        return 0

    clf = get_classifier()
    processed = 0

    for row in logs:
        log_id    = row["log_id"]
        file_path = row["file_path"]

        # Skip if file is missing (device may have sent bad data)
        if not os.path.exists(file_path):
            logger.warning(f"File missing for log_id={log_id}: {file_path}")
            await db.mark_log_processed(log_id)
            processed += 1
            continue

        try:
            result = clf.predict(file_path)
            logger.info(
                f"log_id={log_id}  device={row['device_id']}  "
                f"positive={result['is_positive']}  "
                f"confidence={result['confidence']:.3f}"
            )

            if result["is_positive"]:
                await db.insert_positive(
                    log_id      = log_id,
                    device_id   = row["device_id"],
                    timestamp   = row["timestamp"],
                    confidence  = result["confidence"],
                    latitude    = row["latitude"],
                    longitude   = row["longitude"],
                    label       = f"class_{result['class_idx']}",
                )
            else:
                await db.insert_negative(
                    log_id    = log_id,
                    device_id = row["device_id"],
                    timestamp = row["timestamp"],
                    confidence= result["confidence"],
                )

            await db.mark_log_processed(log_id)
            processed += 1

        except Exception as exc:
            logger.error(f"Error processing log_id={log_id}: {exc}", exc_info=True)
            # Don't mark processed — will retry next cycle

    return processed


# ------------------------------------------------------------------ #

async def main():
    logger.info("=== Forest Guardian Inference Worker ===")
    await db.init_db()

    # Eagerly load model at startup so first batch isn't slow
    logger.info("Loading model …")
    get_classifier()

    logger.info(f"Worker loop started  (interval={WORKER_INTERVAL}s, batch={BATCH_SIZE})")

    while _running:
        try:
            n = await process_batch()
            if n:
                logger.info(f"Batch done: {n} log(s) processed")
        except Exception as exc:
            logger.error(f"Unexpected worker error: {exc}", exc_info=True)

        # Sleep in short increments so SIGTERM is caught quickly
        for _ in range(WORKER_INTERVAL * 10):
            if not _running:
                break
            await asyncio.sleep(0.1)

    await db.close_db()
    logger.info("Worker shut down cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
