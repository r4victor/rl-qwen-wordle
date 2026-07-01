"""Shared Wordle format — one source of truth for training and eval.

Defines the system prompt, the OpenEnv-backed `WordleEnv` (exposing `guess` as a
tool + reward), and the matching tool schema. Training hands `WordleEnv` to TRL;
eval calls `play_episode()`. Both use the same prompt, tool, and reward.
"""

import asyncio
import json
import os

from textarena_env import TextArenaAction, TextArenaEnv

# TRL's environment_factory requires WordleEnv.__init__ to take no args, so the
# server URL is a module global (set by train/eval before the env is built).
_ENV_URL = os.environ.get("WORDLE_ENV_URL", "http://0.0.0.0:8001")


def set_env_url(url: str) -> None:
    global _ENV_URL
    _ENV_URL = url


SYSTEM_PROMPT = """You are an expert Wordle solver with deep knowledge of English vocabulary, letter frequency patterns, and optimal guessing strategies.

Follow these rules to play Wordle:

1. The target is a 5-letter English word
2. You have 6 attempts to guess the correct word
3. After each guess, you receive color-coded feedback:
   - GREEN (G): Letter is correct and in the correct position
   - YELLOW (Y): Letter is in the word but in the wrong position
   - GRAY (X): Letter is not in the word at all
4. All guesses must be valid 5-letter English words
5. You cannot reuse a word you've already guessed
6. Use the tool `guess` to make a guess.
"""


class WordleEnv:
    """OpenEnv-backed Wordle env, per TRL's environment_factory contract:
    a no-arg __init__, reset(), and one tool method (guess).

    The OpenEnv client is async (websocket), so we drive it on a per-instance
    event loop and expose sync methods. The loop is persistent so the session
    established in reset() stays valid across guess() calls.
    """

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self.client = TextArenaEnv(base_url=_ENV_URL)

    def reset(self, **kwargs) -> str | None:
        result = self._loop.run_until_complete(self.client.reset())
        self._last_full_feedback = result.observation.messages[0].content
        self.reward = 0.0
        self.done = False
        return self._last_full_feedback

    def guess(self, guess: str) -> str:
        """
        Make a guess in the Wordle environment.

        Args:
            guess: The guessed word, formatted as '[abcde]'

        Returns:
            The feedback message from the environment.
        """
        if self.done:
            raise ValueError("Game over.")
        result = self._loop.run_until_complete(
            self.client.step(TextArenaAction(message=guess))
        )
        full = result.observation.messages[0].content
        feedback = full[len(self._last_full_feedback):]  # only the new turn
        self._last_full_feedback = full
        self.reward = result.reward  # raw TextArena reward, same on both sides
        self.done = result.done
        return feedback

    def close(self) -> None:
        try:
            self._loop.run_until_complete(self.client.close())
        finally:
            self._loop.close()


# Mirrors WordleEnv.guess(); TRL derives an equivalent schema from the docstring.
GUESS_TOOL = {
    "type": "function",
    "function": {
        "name": "guess",
        "description": "Make a guess in the Wordle environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "guess": {
                    "type": "string",
                    "description": "The guessed word, formatted as '[abcde]'",
                }
            },
            "required": ["guess"],
        },
    },
}


def play_episode(client, model: str, sampling_kwargs: dict | None = None,
                 max_turns: int = 12) -> dict:
    """Play one Wordle episode via OpenAI tool-calling — the format training uses.

    `client` is an openai.OpenAI pointed at the served policy. Returns the
    fields the eval aggregates over.
    """
    sampling_kwargs = sampling_kwargs or {}
    env = WordleEnv()
    try:
        obs = env.reset()
        messages: list = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{obs}"}]

        turns = 0
        while not env.done and turns < max_turns:
            resp = client.chat.completions.create(
                model=model, messages=messages, tools=[GUESS_TOOL], **sampling_kwargs
            )
            msg = resp.choices[0].message
            messages.append(msg)  # assistant turn (may carry tool_calls)
            if not msg.tool_calls:
                break  # model stopped calling the tool
            for tc in msg.tool_calls:
                try:
                    word = json.loads(tc.function.arguments).get("guess", "")
                    result = env.guess(word)
                except ValueError as e:
                    result = str(e)  # "Game over." — teaches the model to stop
                except Exception as e:  # malformed tool args, etc.
                    result = f"Error: {e}"
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )
            turns += 1

        return {"reward": float(env.reward), "solved": env.reward >= 1.0, "turns": turns}
    finally:
        env.close()  # release the OpenEnv session (frees a concurrency slot)
