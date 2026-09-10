"""
main.py — Forest Guardian FastAPI Backend
"""
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

import asyncpg
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Config                                                              #
# ------------------------------------------------------------------ #

AUDIO_DIR        = Path(os.getenv("AUDIO_DIR", "data/audio"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_RECORD_SEC   = int(os.getenv("MAX_RECORD_SEC", "300"))  # 5 min
FRONTEND_ORIGIN  = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
#  App                                                                 #
# ------------------------------------------------------------------ #

app = FastAPI(title="Forest Guardian API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    await db.init_db()
    logger.info("FastAPI started — DB pool ready")


@app.on_event("shutdown")
async def shutdown():
    await db.close_db()


# ------------------------------------------------------------------ #
#  WebSocket manager                                                   #
# ------------------------------------------------------------------ #

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        logger.info(f"WS client connected  ({len(self.active)} total)")

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)
        logger.info(f"WS client disconnected  ({len(self.active)} total)")

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)


manager = ConnectionManager()


# ------------------------------------------------------------------ #
#  Auth helper for devices                                             #
# ------------------------------------------------------------------ #

async def authenticate_device(
    x_device_id: Annotated[str | None, Header()] = None,
    x_api_key:   Annotated[str | None, Header()] = None,
):
    if not x_device_id or not x_api_key:
        raise HTTPException(401, "Missing X-Device-Id or X-Api-Key header")
    ok = await db.verify_device_api_key(x_device_id, x_api_key)
    if not ok:
        raise HTTPException(403, "Invalid device credentials")
    return x_device_id


# ------------------------------------------------------------------ #
#  Schemas                                                             #
# ------------------------------------------------------------------ #

class RecordCommand(BaseModel):
    duration: int = Field(ge=5, le=300, description="Recording duration in seconds (5–300)")


class PositiveOut(BaseModel):
    id: int
    log_id: int
    device_id: str
    device_name: str
    timestamp: datetime
    confidence: float
    latitude: float
    longitude: float
    label: str | None
    file_path: str


# ------------------------------------------------------------------ #
#  Routes — Device side                                                #
# ------------------------------------------------------------------ #

@app.post("/upload", summary="ESP32: Upload audio clip")
async def upload_audio(
    device_id: Annotated[str, Depends(authenticate_device)],
    duration:  Annotated[int | None, Form()] = None,
    audio:     UploadFile = File(...),
):
    """
    Called by the ESP32 device when it has a clip to send.
    Saves the file and creates a log entry.
    """
    # Size guard
    content = await audio.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB > {MAX_FILE_SIZE_MB} MB)")

    # Save file
    ext       = Path(audio.filename).suffix or ".wav"
    fname     = f"{device_id}_{uuid.uuid4().hex}{ext}"
    file_path = AUDIO_DIR / fname
    file_path.write_bytes(content)

    # Insert log
    log_id = await db.insert_log(
        device_id        = device_id,
        file_path        = str(file_path),
        duration_seconds = duration,
        file_size_bytes  = len(content),
    )
    await db.update_device_last_active(device_id)

    logger.info(f"Upload: device={device_id}  log_id={log_id}  size={size_mb:.2f}MB")
    return {"log_id": log_id, "status": "queued"}


@app.get("/device/{device_id}/commands", summary="ESP32: Poll pending commands")
async def poll_commands(
    device_id: str,
    auth: Annotated[str, Depends(authenticate_device)],
):
    """
    Device polls this endpoint to check if there are any pending commands
    (e.g., 'record for 120 seconds').
    """
    commands = await db.get_pending_commands(device_id)
    for cmd in commands:
        await db.mark_command_sent(cmd["id"])
    return {"commands": commands}


# ------------------------------------------------------------------ #
#  Routes — Control room                                               #
# ------------------------------------------------------------------ #

@app.get("/devices", summary="List all devices")
async def list_devices():
    devices = await db.get_all_devices()
    # Don't expose api_key to frontend
    for d in devices:
        d.pop("api_key", None)
        # Convert datetime to ISO string for JSON
        if d.get("last_active"):
            d["last_active"] = d["last_active"].isoformat()
        if d.get("created_at"):
            d["created_at"] = d["created_at"].isoformat()
    return {"devices": devices}


@app.get("/positives", summary="Get positive detections")
async def get_positives(limit: int = 100):
    rows = await db.get_positives(limit)
    for r in rows:
        r["timestamp"]  = r["timestamp"].isoformat()
        r["created_at"] = r["created_at"].isoformat()
    return {"positives": rows}


@app.get("/logs", summary="Get all audio logs")
async def get_logs(limit: int = 50, offset: int = 0):
    rows = await db.get_logs(limit, offset)
    for r in rows:
        r["timestamp"]  = r["timestamp"].isoformat()
        r["created_at"] = r["created_at"].isoformat()
    return {"logs": rows}


@app.get("/audio/{log_id}", summary="Stream audio file for a log")
async def stream_audio(log_id: int):
    log = await db.get_log_by_id(log_id)
    if not log:
        raise HTTPException(404, "Log not found")
    fp = Path(log["file_path"])
    if not fp.exists():
        raise HTTPException(404, "Audio file missing on disk")
    return FileResponse(
        path        = str(fp),
        media_type  = "audio/wav",
        filename    = fp.name,
        headers     = {"Accept-Ranges": "bytes"},
    )


@app.post("/device/{device_id}/record", summary="Send record command to a device")
async def send_record_command(device_id: str, cmd: RecordCommand):
    device = await db.get_device(device_id)
    if not device:
        raise HTTPException(404, "Device not found")

    duration = min(cmd.duration, MAX_RECORD_SEC)
    command_id = await db.insert_command(
        device_id = device_id,
        command   = "record",
        payload   = {"duration": duration},
    )

    logger.info(f"Record command queued: device={device_id}  duration={duration}s  cmd_id={command_id}")
    return {"command_id": command_id, "duration": duration, "status": "pending"}


# ------------------------------------------------------------------ #
#  WebSocket — real-time alerts                                        #
# ------------------------------------------------------------------ #

@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    """
    The React dashboard connects here.
    New positive detections are pushed as JSON events.
    """
    await manager.connect(websocket)
    try:
        # Keep-alive: wait for messages (ping/pong)
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/internal/push-alerts", include_in_schema=False)
async def push_alerts():
    """
    Called by the worker (or a cron) to push unalerted positives
    to connected dashboard clients via WebSocket.
    """
    rows = await db.get_unalerted_positives()
    if not rows:
        return {"pushed": 0}

    for r in rows:
        payload = {
            "type":       "ALERT",
            "id":         r["id"],
            "log_id":     r["log_id"],
            "device_id":  r["device_id"],
            "device_name":r["device_name"],
            "timestamp":  r["timestamp"].isoformat(),
            "confidence": r["confidence"],
            "latitude":   r["latitude"],
            "longitude":  r["longitude"],
            "label":      r["label"],
            "file_path":  r["file_path"],
        }
        await manager.broadcast(payload)

    await db.mark_positives_alerted([r["id"] for r in rows])
    return {"pushed": len(rows)}


# ------------------------------------------------------------------ #
#  Health                                                              #
# ------------------------------------------------------------------ #

@app.get("/health")
async def health():
    return {"status": "ok", "ts": datetime.utcnow().isoformat()}
