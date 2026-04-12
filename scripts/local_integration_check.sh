
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
PORT="${LOCAL_ENV_PORT:-7860}"
ENV_BASE_URL="http://127.0.0.1:${PORT}"
REPORT_PATH="${BASELINE_REPORT_PATH:-artifacts/local_integration_report.json}"
TASKS="${INFERENCE_TASKS:-f1-sprint-dry,f1-feature-safetycar,f1-chaos-weather}"
SEEDS="${INFERENCE_SEEDS:-7}"
MAX_STEPS="${MAX_STEPS:-80}"
FORCE_FALLBACK="${INTEGRATION_FORCE_FALLBACK:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[integration] starting local server on ${ENV_BASE_URL}"
"$PYTHON_BIN" -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT" >/tmp/local_integration_server.log 2>&1 &
SERVER_PID=$!

for _ in {1..60}; do
  if curl -fsS "${ENV_BASE_URL}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

if ! curl -fsS "${ENV_BASE_URL}/health" >/dev/null 2>&1; then
  echo "[integration] server did not become healthy" >&2
  echo "--- server log ---" >&2
  tail -n 120 /tmp/local_integration_server.log >&2 || true
  exit 2
fi

echo "[integration] running inference across tasks=${TASKS} seeds=${SEEDS}"

if [[ "$FORCE_FALLBACK" == "1" ]]; then
  echo "[integration] mode=fallback (keys unset by request)"
  PYTHONPATH=. env \
    -u OPENAI_API_KEY \
    -u HF_TOKEN \
    -u API_KEY \
    ENV_BASE_URL="$ENV_BASE_URL" \
    INFERENCE_TASKS="$TASKS" \
    INFERENCE_SEEDS="$SEEDS" \
    MAX_STEPS="$MAX_STEPS" \
    BASELINE_REPORT_PATH="$REPORT_PATH" \
    "$PYTHON_BIN" inference.py
else
  if [[ -z "${OPENAI_API_KEY:-}" && -z "${HF_TOKEN:-}" && -z "${API_KEY:-}" ]]; then
    echo "[integration] note: no API key found, inference will run in fallback mode"
  else
    echo "[integration] mode=llm (using available API credentials)"
  fi
  PYTHONPATH=. env \
    ENV_BASE_URL="$ENV_BASE_URL" \
    INFERENCE_TASKS="$TASKS" \
    INFERENCE_SEEDS="$SEEDS" \
    MAX_STEPS="$MAX_STEPS" \
    BASELINE_REPORT_PATH="$REPORT_PATH" \
    "$PYTHON_BIN" inference.py
fi

echo "[integration] complete: report at ${REPORT_PATH}"
