#!/usr/bin/env bash
# Launch the Wordle OpenEnv server locally (no hosted HF Space dependency).
#
# GRPO opens one env session per generation, so the server must handle
# concurrency: uvicorn --workers N (N >= generation batch) plus the factory-based
# app (one env instance per WebSocket session) covers it. Trainer connects at
# http://0.0.0.0:8001.
#
#   ./run_env_server.sh            # foreground
#   ./run_env_server.sh &          # background (as used by train.dstack.yml)
set -euo pipefail

export TEXTARENA_ENV_ID="${TEXTARENA_ENV_ID:-Wordle-v0}"
export TEXTARENA_NUM_PLAYERS="${TEXTARENA_NUM_PLAYERS:-1}"
PORT="${PORT:-8001}"
WORKERS="${WORKERS:-16}"

exec python -m uvicorn textarena_env.server.app:app \
  --host 0.0.0.0 --port "$PORT" --workers "$WORKERS"
