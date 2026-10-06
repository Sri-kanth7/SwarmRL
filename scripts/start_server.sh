#!/usr/bin/env bash
# Start the backend (mock state stream on /ws).
# Windows/PowerShell equivalent:
#   python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="${SWARMRL_HOST:-0.0.0.0}"
PORT="${SWARMRL_PORT:-8000}"
python -m uvicorn server.app:app --host "$HOST" --port "$PORT" "$@"
