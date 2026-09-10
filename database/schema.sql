-- ============================================================
--  FOREST GUARDIAN - Database Schema
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ------------------------------------------------------------
-- Table: devices
-- Represents each IoT sensor node in the forest
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS devices (
    device_id   VARCHAR(50) PRIMARY KEY,
    name        VARCHAR(100),
    latitude    DOUBLE PRECISION NOT NULL,
    longitude   DOUBLE PRECISION NOT NULL,
    status      VARCHAR(20) DEFAULT 'active'
                    CHECK (status IN ('active', 'inactive', 'error')),
    last_active TIMESTAMPTZ,
    api_key     VARCHAR(64) UNIQUE NOT NULL DEFAULT gen_random_uuid()::text,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ------------------------------------------------------------
-- Table: logs
-- Every audio file uploaded from any device
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS logs (
    log_id           SERIAL PRIMARY KEY,
    device_id        VARCHAR(50) NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
    timestamp        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    file_path        TEXT NOT NULL,
    duration_seconds INTEGER,
    file_size_bytes  BIGINT,
    processed        BOOLEAN DEFAULT FALSE,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_logs_processed   ON logs(processed);
CREATE INDEX IF NOT EXISTS idx_logs_device_id   ON logs(device_id);
CREATE INDEX IF NOT EXISTS idx_logs_timestamp   ON logs(timestamp DESC);

-- ------------------------------------------------------------
-- Table: positives
-- Logs that triggered a positive detection
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS positives (
    id          SERIAL PRIMARY KEY,
    log_id      INTEGER NOT NULL REFERENCES logs(log_id) ON DELETE CASCADE,
    device_id   VARCHAR(50) NOT NULL REFERENCES devices(device_id),
    timestamp   TIMESTAMPTZ NOT NULL,
    confidence  FLOAT NOT NULL,
    label       VARCHAR(100),
    latitude    DOUBLE PRECISION NOT NULL,
    longitude   DOUBLE PRECISION NOT NULL,
    alerted     BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_positives_alerted   ON positives(alerted);
CREATE INDEX IF NOT EXISTS idx_positives_timestamp ON positives(timestamp DESC);

-- ------------------------------------------------------------
-- Table: negatives
-- Logs that were classified as non-threatening
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS negatives (
    id          SERIAL PRIMARY KEY,
    log_id      INTEGER NOT NULL REFERENCES logs(log_id) ON DELETE CASCADE,
    device_id   VARCHAR(50) NOT NULL REFERENCES devices(device_id),
    timestamp   TIMESTAMPTZ NOT NULL,
    confidence  FLOAT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ------------------------------------------------------------
-- Table: device_commands
-- Commands queued for IoT devices (e.g., record for X seconds)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS device_commands (
    id          SERIAL PRIMARY KEY,
    device_id   VARCHAR(50) NOT NULL REFERENCES devices(device_id),
    command     VARCHAR(50) NOT NULL,
    payload     JSONB,
    status      VARCHAR(20) DEFAULT 'pending'
                    CHECK (status IN ('pending', 'sent', 'executed', 'failed')),
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    executed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_commands_device_status ON device_commands(device_id, status);

-- ------------------------------------------------------------
-- Seed: example devices (remove for production)
-- ------------------------------------------------------------
INSERT INTO devices (device_id, name, latitude, longitude, status, api_key)
VALUES
    ('DEV-001', 'North Perimeter Alpha',  26.1445, 90.6042, 'active', 'key-dev-001-secret'),
    ('DEV-002', 'East Ridge Beta',        26.1523, 90.6187, 'active', 'key-dev-002-secret'),
    ('DEV-003', 'South Valley Gamma',     26.1312, 90.6095, 'active', 'key-dev-003-secret'),
    ('DEV-004', 'West Trail Delta',       26.1401, 90.5921, 'inactive','key-dev-004-secret'),
    ('DEV-005', 'Central Relay Epsilon',  26.1467, 90.6063, 'active', 'key-dev-005-secret')
ON CONFLICT DO NOTHING;
