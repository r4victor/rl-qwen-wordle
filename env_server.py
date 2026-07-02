"""Wordle OpenEnv server with concurrency raised for parallel rollouts.

The stock textarena_env app caps at 1 concurrent session; training and eval open
one session per rollout, so we rebuild the app with a higher max_concurrent_envs.
Run as a SINGLE uvicorn worker — sessions are in-process state, so multiple
workers can't share them.

  MAX_CONCURRENT_ENVS must be >= the generation batch (train --batch-size) and
  the eval --concurrency.
"""

import os
from typing import Any, Optional
from uuid import uuid4

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

    def reset(
        self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any
    ) -> TextArenaObservation:
        # Same as the parent, but forward `seed` to the TextArena env (the parent
        # drops it). A fixed seed → fixed secret word, so a GRPO group can share
        # one word and eval is reproducible.
        env = self._ta_env
        while hasattr(env, "env"):
            if hasattr(env, "full_observations"):
                env.full_observations = {}
            env = env.env
        if hasattr(env, "full_observations"):
            env.full_observations = {}

        self._ta_env.reset(num_players=self.num_players, seed=seed)

        for provider in self._reward_providers:
            provider.reset()
        self._state.episode_id = episode_id if episode_id is not None else str(uuid4())
        self._state.step_count = 0
        self._state.turn = 0
        self._state.last_reward = 0.0
        self._state.last_info = {}
        self._state.raw_state = self._snapshot_state()
        self._last_reward_signals = {}

        observation = self._build_observation()
        observation.reward = 0.0
        observation.done = False
        return observation


def _make_env() -> TextArenaEnvironment:
    return _ConcurrentTextArenaEnvironment(env_id=ENV_ID, num_players=NUM_PLAYERS)


app = create_app(
    _make_env,
    TextArenaAction,
    TextArenaObservation,
    env_name="textarena_env",
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
)
