#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

source "$APP_DIR/.venv/bin/activate"

export DEMO_LOG_LEVEL="${DEMO_LOG_LEVEL:-INFO}"

export MCP_BASE_URL="${MCP_BASE_URL:-http://127.0.0.1:9200}"
export DEPLOY_MAP_PATH="${DEPLOY_MAP_PATH:-$APP_DIR/gateway/deploy-map.json}"
export UPSTREAM_CHAT_PATH="${UPSTREAM_CHAT_PATH:-/chat/completions}"

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-9000}"
export LOG_LEVEL="${LOG_LEVEL:-${DEMO_LOG_LEVEL,,}}"
export LOG_LEVEL="${LOG_LEVEL,,}"


LOG_FILE="${LOG_FILE:-/var/log/kai/gateway.log}"
mkdir -p "$(dirname "$LOG_FILE")"

exec "$APP_DIR/.venv/bin/python" -m gateway.app.main > "$LOG_FILE" 2>&1
