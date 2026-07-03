# rl-qwen-wordle

A small, self-contained example of **RL-training and evaluating** an LLM
(Qwen3.5-4B) to play Wordle, using open-source tools on a single GPU:

- **Env:** [TextArena](https://github.com/TextArena/TextArena) Wordle, served over [OpenEnv](https://github.com/meta-pytorch/OpenEnv)
- **Trainer:** [TRL](https://github.com/huggingface/trl) GRPO (multi-turn, tool-calling)
- **GPU:** provisioned with [dstack](https://github.com/dstackai/dstack)

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

- **Reward:** `1.0` on a win, else `(greens + ½·yellows) / 5` of the last guess (TextArena's own signal).
- **LoRA** is on by default so 4B + vLLM fit one GPU; pass `--no-lora` for full fine-tuning on a bigger box. LoRA pushes a small *adapter* (serve with `ADAPTER=<repo>`); `--no-lora` pushes a *full model* (serve with `MODEL=<repo>`, no adapter).
- vLLM must be started with `--enable-auto-tool-choice --tool-call-parser qwen3_xml` for the tool-calling eval (Qwen3.5 uses XML-style tool calls).
