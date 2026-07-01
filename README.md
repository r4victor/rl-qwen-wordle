# rl-qwen-wordle — OSS Wordle eval/RL

An example of LLM eval and RL training on a Wordle game from [TextArena](https://github.com/TextArena/TextArena).

## Setup

```bash
uv sync
```

## 1. Smoke eval

`smoke_eval.py` plays `NUM_EXAMPLES x ROLLOUTS_PER_EXAMPLE` Wordle games and
reports solve rate / mean reward / mean turns / invalid-move rate. It's
endpoint-agnostic — configured via env vars.

**Eval sizing vs training knobs.** A *game* = one full Wordle episode (= a
*rollout*); an *example* = one seed / secret word. `ROLLOUTS_PER_EXAMPLE`
replays the same word to measure policy variance — the eval analog of
`prime eval run -r` and of RL's `rollouts_per_example`. There is deliberately
**no `batch_size` or `max_steps`** here: those only mean something once you're
computing gradient updates, so they belong to the trainer in step 2, not eval.

Everything is a CLI flag (`python smoke_eval.py --help`). Only `--base-url` and
`--api-key` default from the standard `OPENAI_BASE_URL` / `OPENAI_API_KEY` env
vars, so the key stays out of shell history.

### Option A — OpenRouter (no GPU, quickest harness check)

```bash
OPENAI_API_KEY=$OPENROUTER_API_KEY \
uv run python smoke_eval.py \
  --base-url https://openrouter.ai/api/v1 \
  --model qwen/qwen3.5-4b \
  -n 20
```

### Option B — self-hosted vLLM on a dstack GPU (the real target model)

Bring up the endpoint (forwards to `localhost:8000` while attached):

```bash
dstack apply -f serve-vllm.dstack.yml
```

Then, in another shell:

```bash
uv run python smoke_eval.py \
  --base-url http://localhost:8000/v1 --api-key EMPTY \
  --model Qwen/Qwen3.5-4B --no-enable-thinking \
  -n 20 -r 1
```

### Flags

`--model`, `--base-url`, `--api-key`, `-n/--num-examples`,
`-r/--rollouts-per-example`, `--env-id` (`Wordle-v0` / `-hardcore` / `-long`),
`--max-tokens`, `--temperature`, `--concurrency`,
`--enable-thinking/--no-enable-thinking`, `--log-dir`.

## 2. RL training (next step)

Once the eval baseline looks sane, train with GRPO on the same env. Both stay
fully OSS and run on a dstack GPU:

- **TRL + OpenEnv** — HF-native; TextArena Wordle exposes reset/step, TRL's
  `GRPOTrainer` drives the multi-turn rollout.
- **prime-rl** — self-managed trainer, same `[[env]]` TOML shape as before.

Reward is already RL-ready: `1.0` on solve, shaped in `[0, 1)` otherwise
(fraction of the word revealed).
