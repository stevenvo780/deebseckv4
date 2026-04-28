#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env from project root
if [ -f "../.env" ]; then
  export $(grep -v '^#' ../.env | xargs)
fi

# Install deps if needed
if ! python3 -c "import fastapi, uvicorn, httpx" 2>/dev/null; then
  echo "📦 Instalando dependencias…"
  pip install -q -r requirements.txt
fi

echo ""
echo "  ⚡ NIM Chat arrancando…"
echo "  🌐 http://localhost:8000"
echo ""
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
