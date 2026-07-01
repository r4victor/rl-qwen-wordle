# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "trl==1.7.0",
#   "peft",
#   "vllm",
#   "trackio",
#   "openenv-textarena @ git+https://huggingface.co/spaces/openenv/wordle",
# ]
# ///
"""GRPO training on Wordle with TRL — env + reward come from wordle_env.py.

Based on TRL's first-party example (trl/examples/scripts/openenv/wordle.py),
with Qwen/Qwen3.5-4B + LoRA defaults and sizing flags in plain RL terms.
Needs the Wordle OpenEnv server at --env-url (see run_env_server.sh).
"""

import argparse

from datasets import Dataset
from peft import LoraConfig

from trl import GRPOConfig, GRPOTrainer, RichProgressCallback

# Shared format (prompt + env + guess tool) — identical to what eval_wordle.py uses.
from wordle_env import SYSTEM_PROMPT, WordleEnv, set_env_url


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRPO training for Wordle (TextArena + OpenEnv).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--env-url", default="http://0.0.0.0:8001",
                   help="Wordle OpenEnv server URL (self-host with run_env_server.sh)")
    # --- Sizing knobs ---
    p.add_argument("--group-size", type=int, default=8,
                   help="completions per prompt == GRPO num_generations "
                        "(a.k.a. rollouts_per_example; must be >= 2)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="total completions per optimizer step; "
                        "prompts/step = batch_size / group_size (must divide by group_size)")
    p.add_argument("--steps", type=int, default=10, help="optimizer steps (max_steps)")
    p.add_argument("--micro-batch-size", type=int, default=2,
                   help="completions per forward/backward (GPU-memory knob); "
                        "must divide batch_size. grad-accum = batch_size / micro_batch")
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--max-completion-length", type=int, default=2048,
                   help="TOTAL tokens across all turns of an episode, not per turn")
    p.add_argument("--dataset-size", type=int, default=2000,
                   help="prompt pool size; --steps caps actual steps")
    p.add_argument("--vllm-mode", choices=("colocate", "server"), default="colocate")
    p.add_argument("--vllm-server-url", default="http://localhost:8000")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3,
                   help="colocate shares the GPU with training; keep vLLM's slice small")
    p.add_argument("--lora", default=True, action=argparse.BooleanOptionalAction,
                   help="LoRA (default) fits 4B on one GPU; --no-lora for full fine-tune")
    p.add_argument("--enable-thinking", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--report-to", default="none", choices=("none", "trackio", "wandb"))
    p.add_argument("--push-to-hub", default=None, metavar="REPO_ID",
                   help="after training, push the adapter to this HF repo (needs HF_TOKEN)")
    return p.parse_args()


def reward_func(environments, **kwargs) -> list[float]:
    # Raw TextArena reward, identical to the eval script: 1.0 on win, else a
    # partial score from the last guess -- (greens + 0.5*yellows) / word_length.
    return [env.reward for env in environments]


def main() -> None:
    args = parse_args()
    set_env_url(args.env_url)  # WordleEnv (from wordle_env) reads this

    # batch_size = per_device_batch * grad_accum (single-GPU). TRL needs it
    # divisible by group_size (full groups) and by micro_batch (memory split).
    if args.group_size < 2:
        raise SystemExit("--group-size must be >= 2 (GRPO needs >=2 completions per prompt)")
    if args.batch_size % args.group_size != 0:
        raise SystemExit(
            f"--batch-size ({args.batch_size}) must be divisible by "
            f"--group-size ({args.group_size})"
        )
    if args.batch_size % args.micro_batch_size != 0:
        raise SystemExit(
            f"--batch-size ({args.batch_size}) must be divisible by "
            f"--micro-batch-size ({args.micro_batch_size})"
        )
    grad_accum = args.batch_size // args.micro_batch_size
    prompts_per_step = args.batch_size // args.group_size
    print(
        f"[sizing] group_size={args.group_size}  batch_size={args.batch_size}  "
        f"steps={args.steps}  ->  {prompts_per_step} prompts/step, "
        f"micro_batch={args.micro_batch_size} x grad_accum={grad_accum}"
    )

    # TRL drives WordleEnv's guess() tool over the multi-turn loop automatically.
    output_dir = args.output_dir or f"outputs/{args.model.split('/')[-1]}-wordle-GRPO"
    dataset = Dataset.from_dict(
        {"prompt": [[{"role": "user", "content": SYSTEM_PROMPT}] for _ in range(args.dataset_size)]}
    )

    peft_config = None
    if args.lora:
        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules="all-linear", task_type="CAUSAL_LM",
        )

    config = GRPOConfig(
        output_dir=output_dir,
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_url if args.vllm_mode == "server" else None,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        num_generations=args.group_size,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=grad_accum,
        max_steps=args.steps,
        learning_rate=args.learning_rate,
        max_completion_length=args.max_completion_length,
        chat_template_kwargs={"enable_thinking": args.enable_thinking},
        # Log reward/kl/loss + sample games to stdout every step.
        log_completions=True,
        num_completions_to_print=2,
        logging_steps=1,
        report_to=args.report_to,
        hub_model_id=args.push_to_hub,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=config,
        peft_config=peft_config,
        environment_factory=WordleEnv,
        callbacks=[RichProgressCallback()],
    )
    trainer.train()
    trainer.save_model(output_dir)
    if args.push_to_hub:
        trainer.push_to_hub()  # uploads the adapter to hub_model_id


if __name__ == "__main__":
    main()
