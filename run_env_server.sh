#!/usr/bin/env bash
# Launch the Wordle OpenEnv server locally (env_server.py raises the concurrency
# cap so parallel rollouts don't hit "Server at capacity"). Single worker —
# sessions are in-process state. Trainer/eval connect at http://0.0.0.0:8001.
#
#   bash run_env_server.sh            # foreground
#   bash run_env_server.sh &          # background
set -euo pipefail

export TEXTARENA_ENV_ID="${TEXTARENA_ENV_ID:-Wordle-v0}"
export TEXTARENA_NUM_PLAYERS="${TEXTARENA_NUM_PLAYERS:-1}"
export MAX_CONCURRENT_ENVS="${MAX_CONCURRENT_ENVS:-64}"
PORT="${PORT:-8001}"

exec python -m uvicorn env_server:app --host 0.0.0.0 --port "$PORT"
