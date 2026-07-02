# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "openai",
#   "openenv-textarena @ git+https://huggingface.co/spaces/openenv/wordle",
# ]
# ///
"""Wordle eval — same format as training (tool-calling + OpenEnv).

Requires the Wordle OpenEnv server running (run_env_server.sh) and an
OpenAI-compatible endpoint serving the model.

  # baseline (base model) then trained (LoRA served under name "wordle"):
  uv run eval_wordle.py --base-url http://localhost:8000/v1 --model Qwen/Qwen3.5-4B -n 50
  uv run eval_wordle.py --base-url http://localhost:8000/v1 --model wordle -n 50
"""

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from wordle_env import play_episode, set_env_url


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Eval a Wordle policy in the same tool-calling format as training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model", required=True, help="model/adapter name the endpoint serves"
    )
    p.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible endpoint serving the policy",
    )
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument(
        "--env-url",
        default="http://0.0.0.0:8001",
        help="Wordle OpenEnv server (run_env_server.sh)",
    )
    p.add_argument("-n", "--num-examples", type=int, default=20)
    p.add_argument("-r", "--rollouts-per-example", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument(
        "--max-tokens", type=int, default=1024, help="per-turn completion cap"
    )
    p.add_argument("--max-turns", type=int, default=12, help="safety cap on tool calls")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument(
        "--enable-thinking", default=None, action=argparse.BooleanOptionalAction
    )
    p.add_argument("--log-dir", default="logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_env_url(args.env_url)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    sampling = {
        "temperature": args.temperature,
        "max_completion_tokens": args.max_tokens,
    }
    if args.enable_thinking is not None:
        sampling["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": args.enable_thinking}
        }

    total = args.num_examples * args.rollouts_per_example
    print(f"Endpoint : {args.base_url}")
    print(f"Model    : {args.model}   (tool-calling, same format as training)")
    print(f"Games    : {args.num_examples} x {args.rollouts_per_example} = {total}\n")

    # Seed by example index so the -r replays of an example share a word, and
    # baseline vs trained runs face identical words (paired comparison).
    def run_one(i):
        seed = i // args.rollouts_per_example
        return play_episode(client, args.model, sampling, max_turns=args.max_turns, seed=seed)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(run_one, range(total)))

    for i, r in enumerate(results):
        tag = "WIN " if r["solved"] else "----"
        print(f"  [{tag}] game={i:>3}  reward={r['reward']:.3f}  turns={r['turns']}")

    solved = sum(r["solved"] for r in results)
    summary = {
        "solve_rate": solved / total,
        "mean_reward": statistics.mean(r["reward"] for r in results),
        "mean_turns": statistics.mean(r["turns"] for r in results),
        "solved": solved,
        "total": total,
    }
    print("\n=== summary ===")
    print(f"solve rate  : {solved}/{total} = {summary['solve_rate']:.1%}")
    print(f"mean reward : {summary['mean_reward']:.3f}")
    print(f"mean turns  : {summary['mean_turns']:.2f}")

    write_run_log(args, results, summary)


def write_run_log(args: argparse.Namespace, results: list[dict], summary: dict) -> None:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_model = args.model.replace("/", "--")
    run_dir = Path(args.log_dir) / f"{stamp}_wordle-toolcall_{safe_model}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    config = {
        "timestamp": stamp,
        "format": "tool-calling",
        "model": args.model,
        "base_url": args.base_url,
        "env_url": args.env_url,
        "num_examples": args.num_examples,
        "rollouts_per_example": args.rollouts_per_example,
        "temperature": args.temperature,
        "enable_thinking": args.enable_thinking,
    }
    with (run_dir / "summary.json").open("w") as f:
        json.dump({"config": config, "summary": summary}, f, indent=2)
    print(f"\nlogged to {run_dir}")


if __name__ == "__main__":
    main()
