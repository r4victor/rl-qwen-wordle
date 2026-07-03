# rl-qwen-wordle

An example of **RL-training and evaluating** an LLM (Qwen3.5-4B) to play Wordle, using open-source tools on a single GPU:

- **Env:** [TextArena](https://github.com/TextArena/TextArena) Wordle, served over [OpenEnv](https://github.com/meta-pytorch/OpenEnv)
- **Trainer:** [TRL](https://github.com/huggingface/trl) GRPO (multi-turn, tool-calling)
- **GPU:** provisioned with [dstack](https://github.com/dstackai/dstack)

Implements both LoRA and full training.

## Results

* Baseline Qwen3.5-4B with thinking enabled and 1024 tokens/turn scores **15-20%** win rate.
* A LoRA training (batch_size=32, steps=~30) improves win rate up to **30%** but increasing the number of steps doesn't help much.
* A full training (batch_size=32, steps=~30) shows the same improvement but continues to improve with the number of steps. (Haven't measure >40 steps).
* The TextArena default reward is tweaked to give bonus to valid words and penalize guesses after the game end. With the default reward that scores only % of guessed letters, the model constantly tries non-existent words.
* The eval caps 1024 tokens/turn, while training caps 2048 tokens/game. 2048 tokens/game seems to be low since the model cannot finish most games due to token budget. It's TBD to try larger `--max-completion-length` (requires >80GB VRAM) and/or disable thinking.

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
`--steps 10`. Watch **reward** climb via `dstack logs wordle-grpo`. When training
finishes, the run **serves and evals the trained model** — the numbers print at
the end of the log.

The adapter is saved to `outputs/` on the run — which is ephemeral. To keep it
(and serve it in a separate run), **push it to HF Hub**: set `HF_REPO` and
`HF_TOKEN` in `train.dstack.yml` and it uploads the adapter when training ends.

## Serve & eval separately

To eval later (or a model pushed to HF), serve the base model or a trained
adapter with `serve-vllm.dstack.yml`:

```bash
dstack apply -f serve-vllm.dstack.yml                                    # base (baseline)
ADAPTER=your-user/qwen35-4b-wordle dstack apply -f serve-vllm.dstack.yml # trained (from HF)
```

It forwards port 8000 to localhost. Start the env server and run the eval
against it (`--model wordle` for a served adapter, else the base model id):

```bash
bash run_env_server.sh &
uv run eval_wordle.py --base-url http://localhost:8000/v1 --model wordle -n 50
```

## Notes

- **LoRA** is on by default so 4B + vLLM fit one GPU; pass `--no-lora` for full fine-tuning on a bigger box. LoRA pushes a small *adapter* (serve with `ADAPTER=<repo>`); `--no-lora` pushes a *full model* (serve with `MODEL=<repo>`, no adapter).
