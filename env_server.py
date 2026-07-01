"""Wordle OpenEnv server with concurrency raised for parallel rollouts.

The stock textarena_env app caps at 1 concurrent session; training and eval open
one session per rollout, so we rebuild the app with a higher max_concurrent_envs.
Run as a SINGLE uvicorn worker — sessions are in-process state, so multiple
workers can't share them.

  MAX_CONCURRENT_ENVS must be >= the generation batch (train --batch-size) and
  the eval --concurrency.
"""

import os

from openenv.core.env_server.http_server import create_app
from textarena_env.models import TextArenaAction, TextArenaObservation
from textarena_env.server.environment import TextArenaEnvironment

ENV_ID = os.getenv("TEXTARENA_ENV_ID", "Wordle-v0")
NUM_PLAYERS = int(os.getenv("TEXTARENA_NUM_PLAYERS", "1"))
MAX_CONCURRENT_ENVS = int(os.getenv("MAX_CONCURRENT_ENVS", "64"))


class _ConcurrentTextArenaEnvironment(TextArenaEnvironment):
    # The factory below builds a fresh instance per WebSocket session, so state
    # is isolated — safe to opt into openenv-core's concurrent-session support.
    SUPPORTS_CONCURRENT_SESSIONS = True


def _make_env() -> TextArenaEnvironment:
    return _ConcurrentTextArenaEnvironment(env_id=ENV_ID, num_players=NUM_PLAYERS)


app = create_app(
    _make_env,
    TextArenaAction,
    TextArenaObservation,
    env_name="textarena_env",
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
)
