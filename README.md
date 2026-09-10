# 🌲 Forest Guardian — Acoustic Threat Detection System

A full-stack IoT monitoring platform for detecting suspicious sounds in forest environments.
ESP32 sensor nodes stream raw audio to a central server where a PyTorch model classifies
each clip in real-time. Alerts appear instantly on a tactical map dashboard.

---

## Architecture

```
ESP32 Nodes (forest)
      │  WiFi / HTTP POST /upload
      ▼
FastAPI Backend  ──────────────────────────────┐
  ├─ /upload          (device → server)        │
  ├─ /device/{id}/commands (device polls)      │
  ├─ /positives, /logs, /devices (dashboard)  │ asyncpg
  ├─ /audio/{log_id}  (audio stream)           │
  ├─ POST /device/{id}/record (ctrl room)      │
  └─ WebSocket /ws/alerts (push alerts)        ▼
                                         PostgreSQL DB
Inference Worker (separate process)       ├─ devices
  └─ polls logs → PyTorch → positives/    ├─ logs
                             negatives    ├─ positives
                                          ├─ negatives
React Dashboard (Leaflet + OpenStreetMap) └─ device_commands
  ├─ Map with device markers (green)
  ├─ Alert markers (pulsing red)
  ├─ Alerts panel + audio playback
  ├─ Logs panel + audio playback
  └─ Devices panel + record command modal
```

---

## Project Structure

```
forest-guardian/
│
├── backend/
│   ├── main.py          ← FastAPI app (all HTTP + WebSocket routes)
│   ├── db.py            ← asyncpg pool + all query helpers
│   ├── model.py         ← PyTorch inference wrapper
│   ├── worker.py        ← Background inference loop (run separately)
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx           ← Full dashboard layout
│   │   ├── main.jsx          ← React entry point
│   │   ├── index.css         ← Tactical dark theme
│   │   ├── api/client.js     ← API + WebSocket client
│   │   ├── hooks/useAlerts.js← WebSocket hook with auto-reconnect
│   │   └── components/
│   │       ├── ForestMap.jsx ← Leaflet map (device + alert markers)
│   │       └── AudioPlayer.jsx ← Custom audio player
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   └── .env.example
│
├── database/
│   └── schema.sql       ← Full PostgreSQL schema + seed data
│
└── models/
    └── model.pth        ← ← ← PUT YOUR TRAINED MODEL HERE
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 14+
- Your trained PyTorch model (`.pth` file)

---

### 2. Database

```bash
# Create database
createdb forest_monitor

# Apply schema
psql -d forest_monitor -f database/schema.sql
```

---

### 3. Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and edit config
cp .env.example .env
# Edit .env — set DATABASE_URL, MODEL_PATH, etc.

# Start API server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# In a separate terminal, start the inference worker
python worker.py
```

---

### 4. Frontend

```bash
cd frontend

npm install

# Copy and edit config
cp .env.example .env.local
# Set VITE_API_URL=http://localhost:8000

npm run dev
# → http://localhost:5173
```

---

### 5. Place Your Model

Copy your trained `.pth` file to `models/model.pth`

The model must accept input shape `(B, 1, N_MELS, T)` and output `(B, num_classes)` logits.

If your preprocessing differs, edit the `preprocess()` method in `backend/model.py`.

Key env vars for model tuning:

| Variable                  | Default | Meaning                              |
|---------------------------|---------|--------------------------------------|
| `MODEL_PATH`              | `../models/model.pth` | Path to .pth file        |
| `MODEL_SAMPLE_RATE`       | `16000` | Target sample rate (Hz)              |
| `MODEL_N_MELS`            | `128`   | Number of mel filterbanks            |
| `MODEL_N_FFT`             | `1024`  | FFT window size                      |
| `MODEL_HOP_LENGTH`        | `512`   | Hop length                           |
| `MODEL_POSITIVE_CLASS_IDX`| `1`     | Index of the "threat" class          |
| `MODEL_MAX_DURATION_SEC`  | `60`    | Max audio duration (pad/trim to this)|

---

## ESP32 Integration

### Upload audio clip

```
POST http://<server>:8000/upload
Headers:
  X-Device-Id: DEV-001
  X-Api-Key:   key-dev-001-secret
Form:
  audio:    <.wav file>
  duration: 60
```

### Poll for commands

```
GET http://<server>:8000/device/DEV-001/commands
Headers:
  X-Device-Id: DEV-001
  X-Api-Key:   key-dev-001-secret
```

Response when a record command is pending:
```json
{
  "commands": [
    { "command": "record", "payload": { "duration": 120 } }
  ]
}
```

The ESP32 should poll this endpoint every 10–30 seconds.

---

## API Reference

| Method | Path                          | Description                          |
|--------|-------------------------------|--------------------------------------|
| POST   | `/upload`                     | Device: upload audio clip            |
| GET    | `/device/{id}/commands`       | Device: poll pending commands        |
| GET    | `/devices`                    | Dashboard: list all devices          |
| GET    | `/positives?limit=N`          | Dashboard: get detections            |
| GET    | `/logs?limit=N&offset=N`      | Dashboard: get all audio logs        |
| GET    | `/audio/{log_id}`             | Dashboard: stream audio file         |
| POST   | `/device/{id}/record`         | Dashboard: queue record command      |
| WS     | `/ws/alerts`                  | Dashboard: real-time alert stream    |
| POST   | `/internal/push-alerts`       | Worker: push new positives to WS     |
| GET    | `/health`                     | Health check                         |

---

## Dashboard Features

| Feature | How it works |
|---|---|
| Map with device nodes | Leaflet + OpenStreetMap, green dots for active devices |
| Pulsing red alert markers | Click any red dot to open audio + detail panel |
| Real-time alerts | WebSocket push from server → instant map update |
| Audio playback | Custom player with scrubber — works for any log |
| Request recording | Nodes tab → click REC → slider for duration (5–300s) |
| Live/reconnecting indicator | Top bar shows WebSocket status |
| Auto-poll fallback | Every 10 s even without WebSocket |

---

## Alert Push Flow

The worker calls `POST /internal/push-alerts` after each inference batch.
This route reads all unalerted positives, broadcasts them via WebSocket to every
connected dashboard, then marks them as alerted.

To call it automatically from the worker, add at the end of `process_batch()`:

```python
import aiohttp
async with aiohttp.ClientSession() as s:
    await s.post("http://localhost:8000/internal/push-alerts")
```

Or just rely on the 10-second dashboard poll — the alerts will appear either way.

---

## Adding a New Device

```sql
INSERT INTO devices (device_id, name, latitude, longitude, api_key)
VALUES ('DEV-006', 'New Node Alpha', 26.1508, 90.6100, 'your-secret-key-here');
```

---

## Production Notes

- Run with `uvicorn main:app --workers 4` behind an nginx reverse proxy
- Run `worker.py` as a systemd service
- Use environment variables (never hardcode DB credentials)
- Rate-limit the `/upload` endpoint per device
- Consider S3/MinIO for audio storage at scale
- Add HTTPS (Let's Encrypt) before deploying remotely
