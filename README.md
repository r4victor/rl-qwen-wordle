# rl-qwen-wordle

A small, self-contained example of **RL-training and evaluating** an LLM
(Qwen3.5-4B) to play Wordle, using open-source tools on a single GPU:

- **Env:** [TextArena](https://github.com/TextArena/TextArena) Wordle, served over [OpenEnv](https://github.com/meta-pytorch/OpenEnv)
- **Trainer:** [TRL](https://github.com/huggingface/trl) GRPO (multi-turn, tool-calling)
- **GPU:** provisioned with [dstack](https://github.com/dstackai/dstack)

Inspired by Prime Intellect's hosted-training example, but does not depend on it.

## Files

| File | What it is |
|------|-----------|
| `wordle_env.py` | The shared format — system prompt + `guess` tool + reward. Imported by **both** training and eval, so they measure the same thing. |
| `train_grpo.py` | GRPO training with TRL. |
| `eval_wordle.py` | Eval — plays Wordle in the same tool-calling format. |
| `run_env_server.sh` | Starts the Wordle OpenEnv server. |
| `train.dstack.yml` | Runs training on a GPU. |
| `serve-vllm.dstack.yml` | Serves a model on a GPU (for eval). |

## Train

```bash
dstack apply -f train.dstack.yml
```

Runs the OpenEnv server + GRPO training (vLLM colocate, LoRA) on one GPU. Main
knobs: `--group-size 8` (completions per prompt), `--batch-size 16`,
`--steps 10`. Watch **reward** climb via `dstack logs wordle-grpo`. The trained
LoRA adapter is saved to `outputs/<model>-wordle-GRPO/`.

## Eval

Serve the model (tool-calling enabled) and the env server, then run the eval:

```bash
vllm serve Qwen/Qwen3.5-4B --port 8000 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --enable-lora --lora-modules wordle=outputs/Qwen3.5-4B-wordle-GRPO --max-lora-rank 16 &
bash run_env_server.sh &

uv run eval_wordle.py --base-url http://localhost:8000/v1 --model Qwen/Qwen3.5-4B -n 50  # baseline
uv run eval_wordle.py --base-url http://localhost:8000/v1 --model wordle          -n 50  # trained
```

Solve rate / mean reward print to the console and are saved to `logs/` (git-ignored).

## Notes

- **Reward:** `1.0` on a win, else `(greens + ½·yellows) / 5` of the last guess (TextArena's own signal).
- **LoRA** is on by default so 4B + vLLM fit one GPU; pass `--no-lora` for full fine-tuning on a bigger box.
- vLLM must be started with `--enable-auto-tool-choice --tool-call-parser hermes` for the tool-calling eval.
