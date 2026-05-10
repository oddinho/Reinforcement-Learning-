"""Sweep v5 cluster radii from the final v3 checkpoint.

This tests two changes together:

1. append opponent absolute board position to actor features;
2. vary the local and large BFS cluster radii.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = EXPERIMENT_ROOT / "checkpoints"
LOG_DIR = EXPERIMENT_ROOT / "training_logs"
DEFAULT_RESUME_PATH = (
    EXPERIMENT_ROOT.parent / "deeprl_v3" / "checkpoints" / "ppo_league_gen5.pt"
)
SUMMARY_PATH = EXPERIMENT_ROOT / "cluster_radius_sweep_summary.json"

MEAN_RE = re.compile(
    r"eval opponent=(\S+).*mean_score_diff=([-0-9.]+).*min=([-0-9]+).*max=([-0-9]+)"
)
SEED_RE = re.compile(r"eval_seed seed=(\d+).*score_diff=([-0-9]+)")


def run_logged(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=EXPERIMENT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {log_path}")


def parse_eval_log(path):
    means = {}
    mins = {}
    maxs = {}
    wins = {}
    current_opponent = None
    for line in path.read_text(encoding="utf-8").splitlines():
        mean_match = MEAN_RE.search(line)
        if mean_match:
            current_opponent = mean_match.group(1)
            means[current_opponent] = float(mean_match.group(2))
            mins[current_opponent] = int(mean_match.group(3))
            maxs[current_opponent] = int(mean_match.group(4))
            wins[current_opponent] = 0
            continue
        seed_match = SEED_RE.search(line)
        if seed_match and current_opponent is not None:
            score_diff = int(seed_match.group(2))
            if score_diff > 0:
                wins[current_opponent] += 1
    return {"means": means, "mins": mins, "maxs": maxs, "wins": wins}


def selection_tuple(parsed, metric_opponents):
    means = parsed["means"]
    wins = parsed["wins"]
    primary_min = min(means[name] for name in metric_opponents)
    all_min = min(means.values())
    all_mean = sum(means.values()) / max(1, len(means))
    min_wins = min(wins.values())
    return (primary_min, all_min, all_mean, min_wins)


def run_combo(args, local_radius, large_radius, index):
    base_token = f"l{local_radius}_g{large_radius}"
    token = f"{args.tag}_{base_token}" if args.tag else base_token
    checkpoint = CHECKPOINT_DIR / f"sweep_radius_{token}_latest.pt"
    best_checkpoint = checkpoint.with_name(checkpoint.stem + "_best.pt")
    train_log = LOG_DIR / f"sweep_radius_{token}_train.log"
    eval_log = LOG_DIR / f"sweep_radius_{token}_eval.log"
    seed = args.seed + index * args.seed_stride

    train_command = [
        sys.executable,
        "train_target_selection_ppo_v5.py",
        "--seed",
        str(seed),
        "--resume-path",
        str(args.resume_path),
        "--iterations",
        str(args.iterations),
        "--rollout-steps",
        str(args.rollout_steps),
        "--ppo-epochs",
        str(args.ppo_epochs),
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--learning-rate",
        str(args.learning_rate),
        "--entropy-coef",
        str(args.entropy_coef),
        "--terminal-win-bonus",
        str(args.terminal_win_bonus),
        "--reward-mode",
        args.reward_mode,
        "--local-cluster-radius",
        str(local_radius),
        "--large-cluster-radius",
        str(large_radius),
        "--opponent-mix",
        args.opponent_mix,
        "--eval-opponents",
        *args.eval_opponents,
        "--selection-opponents",
        *args.selection_opponents,
        "--eval-seeds",
        *[str(seed_value) for seed_value in args.train_eval_seeds],
        "--checkpoint-path",
        str(checkpoint),
    ]
    run_logged(train_command, train_log)

    eval_command = [
        sys.executable,
        "train_target_selection_ppo_v5.py",
        "--eval-only",
        "--checkpoint-path",
        str(best_checkpoint),
        "--eval-opponents",
        *args.eval_opponents,
        "--eval-seeds",
        *[str(seed_value) for seed_value in args.eval_seeds],
        "--eval-steps",
        str(args.eval_steps),
        "--local-cluster-radius",
        str(local_radius),
        "--large-cluster-radius",
        str(large_radius),
    ]
    run_logged(eval_command, eval_log)
    parsed = parse_eval_log(eval_log)
    missing = [name for name in args.eval_opponents if name not in parsed["means"]]
    if missing:
        raise RuntimeError(f"Missing eval results for {missing} in {eval_log}")

    return {
        "local_radius": local_radius,
        "large_radius": large_radius,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "train_log": str(train_log),
        "eval_log": str(eval_log),
        **parsed,
        "selection_tuple": selection_tuple(parsed, args.metric_opponents),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-radii", nargs="+", type=int, default=[3, 4])
    parser.add_argument("--large-radii", nargs="+", type=int, default=[5, 6, 7])
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--seed-stride", type=int, default=100)
    parser.add_argument("--tag", default="")
    parser.add_argument("--resume-path", type=Path, default=DEFAULT_RESUME_PATH)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--rollout-steps", type=int, default=1500)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--terminal-win-bonus", type=float, default=5.0)
    parser.add_argument("--reward-mode", default="score_delta_cluster")
    parser.add_argument(
        "--opponent-mix",
        default="bfs:0.15,rollout_gated:0.25,ppo:0.25,position_aware_ppo:0.35",
    )
    parser.add_argument(
        "--eval-opponents",
        nargs="+",
        default=["bfs", "rollout_gated", "ppo", "position_aware_ppo"],
    )
    parser.add_argument(
        "--selection-opponents",
        nargs="+",
        default=["bfs", "rollout_gated", "ppo", "position_aware_ppo"],
    )
    parser.add_argument(
        "--metric-opponents",
        nargs="+",
        default=["rollout_gated", "position_aware_ppo"],
    )
    parser.add_argument("--train-eval-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[1000, 1001, 1002])
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--summary-path", type=Path, default=SUMMARY_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    results = []
    index = 0
    for local_radius in args.local_radii:
        for large_radius in args.large_radii:
            if large_radius < local_radius:
                continue
            result = run_combo(args, local_radius, large_radius, index)
            results.append(result)
            index += 1
            print(
                f"finished local={local_radius} large={large_radius} "
                f"selection={result['selection_tuple']} means={result['means']}",
                flush=True,
            )

    ranked = sorted(results, key=lambda row: row["selection_tuple"], reverse=True)
    summary = {
        "resume_path": str(args.resume_path),
        "iterations": args.iterations,
        "rollout_steps": args.rollout_steps,
        "train_eval_seeds": args.train_eval_seeds,
        "eval_seeds": args.eval_seeds,
        "eval_opponents": args.eval_opponents,
        "metric_opponents": args.metric_opponents,
        "results": results,
        "ranked": ranked,
        "best": ranked[0] if ranked else None,
    }
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if ranked:
        best = ranked[0]
        print(
            f"best local={best['local_radius']} large={best['large_radius']} "
            f"selection={best['selection_tuple']} means={best['means']}",
            flush=True,
        )
    print(f"summary={args.summary_path}", flush=True)


if __name__ == "__main__":
    main()
