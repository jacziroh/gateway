#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"


source "$APP_DIR/.venv/bin/activate"


export DEMO_LOG_LEVEL="${DEMO_LOG_LEVEL:-INFO}"
export MCP_LOG_LEVEL="${MCP_LOG_LEVEL:-${DEMO_LOG_LEVEL,,}}"


export MCP_HOST="${MCP_HOST:-0.0.0.0}"
export MCP_PORT="${MCP_PORT:-9200}"


LOG_FILE="${LOG_FILE:-/var/log/kai/mcp.log}"
mkdir -p "$(dirname "$LOG_FILE")"


exec "$APP_DIR/.venv/bin/python" -m mcp.app.main > "$LOG_FILE" 2>&1

