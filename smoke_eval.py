"""OSS Wordle smoke eval — TextArena env + any OpenAI-compatible endpoint.

The same script works against:
  - OpenRouter (hosted OSS model, no GPU) for a quick harness check, or
  - a self-hosted vLLM (e.g. Qwen/Qwen3.5-4B on a dstack GPU).

Eval sizing mirrors `prime eval run -n -r` (and the RL `rollouts_per_example`):
a game = one full Wordle episode; an "example" = one seed / secret word;
--rollouts-per-example replays of the same word estimate policy variance.
Total games played = num_examples * rollouts_per_example. (There is no
`batch_size`/`max_steps` here — those only exist once you're training.)

Run `python smoke_eval.py --help` for all flags.
"""

import argparse
import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import textarena as ta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Play Wordle with an OpenAI-compatible model and report metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="openai/gpt-4o-mini",
                   help="model name as the endpoint expects it")
    # Endpoint + key default from the standard OpenAI SDK env vars so secrets
    # stay out of shell history; override with the flags when convenient.
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"),
                   help="e.g. https://openrouter.ai/api/v1 or http://localhost:8000/v1 "
                        "[env: OPENAI_BASE_URL]")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                   help='endpoint key ("EMPTY" is fine for a local vLLM) [env: OPENAI_API_KEY]')
    p.add_argument("-n", "--num-examples", type=int, default=20,
                   help="distinct secret words / puzzles")
    p.add_argument("-r", "--rollouts-per-example", type=int, default=1,
                   help="replays per word (>1 to measure policy variance)")
    p.add_argument("--env-id", default="Wordle-v0",
                   help="TextArena env id: Wordle-v0 / -hardcore / -long")
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="per-turn completion cap")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--concurrency", type=int, default=8,
                   help="games played in parallel")
    p.add_argument("--enable-thinking", default=None, action=argparse.BooleanOptionalAction,
                   help="toggle Qwen/vLLM thinking mode (default: leave to the model)")
    p.add_argument("--log-dir", default="logs",
                   help="write per-run results.jsonl + summary.json here (git-ignored)")
    return p.parse_args()


def play_one_game(args: argparse.Namespace, example_id: int, rollout_idx: int) -> dict:
    # The seed selects the secret word, so a fixed example_id => same puzzle,
    # replayed rollout_idx times (temperature drives the variation between them).
    seed = example_id

    extra_kwargs = {}
    if args.enable_thinking is not None:
        extra_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": args.enable_thinking}
        }

    agent = ta.agents.OpenAIAgent(
        model_name=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_completion_tokens=args.max_tokens,
        temperature=args.temperature,
        **extra_kwargs,
    )
    env = ta.make(env_id=args.env_id)
    env.reset(num_players=1, seed=seed)

    done = False
    turns = 0
    while not done:
        _, observation = env.get_observation()
        action = agent(observation)
        done, _ = env.step(action=action)
        turns += 1

    rewards, game_info = env.close()
    reward = float(rewards[0]) if rewards and 0 in rewards else 0.0
    info = game_info.get(0, {})
    return {
        "example_id": example_id,
        "rollout_idx": rollout_idx,
        "reward": reward,
        "solved": reward >= 1.0,
        "turns": turns,
        "invalid_move": bool(info.get("invalid_move", False)),
        "reason": info.get("reason", ""),
    }


def main() -> None:
    args = parse_args()
    n_ex, n_roll = args.num_examples, args.rollouts_per_example
    total = n_ex * n_roll
    print(f"Endpoint : {args.base_url}")
    print(f"Model    : {args.model}")
    print(
        f"Env      : {args.env_id}  |  examples={n_ex} x rollouts={n_roll} "
        f"= {total} games  temp={args.temperature}  max_tokens={args.max_tokens}\n"
    )

    tasks = [(ex, r) for ex in range(n_ex) for r in range(n_roll)]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda t: play_one_game(args, *t), tasks))

    for r in results:
        tag = "WIN " if r["solved"] else "----"
        note = "" if not r["invalid_move"] else f"  invalid: {r['reason']}"
        print(
            f"  [{tag}] ex={r['example_id']:>3} roll={r['rollout_idx']}  "
            f"reward={r['reward']:.3f}  turns={r['turns']}{note}"
        )

    solved = sum(r["solved"] for r in results)
    invalid = sum(r["invalid_move"] for r in results)
    summary = {
        "solve_rate": solved / total,
        "mean_reward": statistics.mean(r["reward"] for r in results),
        "mean_turns": statistics.mean(r["turns"] for r in results),
        "invalid_rate": invalid / total,
        "solved": solved,
        "total": total,
    }
    print("\n=== summary ===")
    print(f"solve rate    : {solved}/{total} = {summary['solve_rate']:.1%}")
    print(f"mean reward   : {summary['mean_reward']:.3f}")
    print(f"mean turns    : {summary['mean_turns']:.2f}")
    print(f"invalid moves : {invalid}/{total} = {summary['invalid_rate']:.1%}")

    # Per-example variance is the reason to replay a word more than once.
    if n_roll > 1:
        summary["per_example_solve"] = statistics.mean(
            statistics.mean(r["solved"] for r in results if r["example_id"] == ex)
            for ex in range(n_ex)
        )
        print(f"per-example solve (avg): {summary['per_example_solve']:.1%}")

    write_run_log(args, results, summary)


def write_run_log(args: argparse.Namespace, results: list[dict], summary: dict) -> None:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_model = args.model.replace("/", "--")
    run_dir = Path(args.log_dir) / f"{stamp}_{args.env_id}_{safe_model}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    config = {
        "timestamp": stamp,
        "model": args.model,
        "base_url": args.base_url,
        "env_id": args.env_id,
        "num_examples": args.num_examples,
        "rollouts_per_example": args.rollouts_per_example,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "enable_thinking": args.enable_thinking,
    }
    with (run_dir / "summary.json").open("w") as f:
        json.dump({"config": config, "summary": summary}, f, indent=2)

    print(f"\nlogged to {run_dir}")


if __name__ == "__main__":
    main()
