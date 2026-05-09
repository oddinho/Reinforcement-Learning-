"""Short cluster-shaping cap sweep, followed by a v3 league run.

This script is intentionally small and reproducible:

1. warm-start from the position-aware PPO checkpoint,
2. train one short run for each cluster-signal clipping cap,
3. evaluate each best checkpoint,
4. choose the cap with the best conservative score,
5. start the normal league loop with that cap and shorter finetune iterations.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = EXPERIMENT_ROOT / "checkpoints"
LOG_DIR = EXPERIMENT_ROOT / "training_logs"
DEFAULT_RESUME_PATH = (
    EXPERIMENT_ROOT.parent
    / "checkpoints"
    / "target_selection_ppo_mlp_position_selfplay_latest_best.pt"
)
SUMMARY_PATH = EXPERIMENT_ROOT / "cluster_cap_sweep_summary.json"

MEAN_RE = re.compile(r"eval opponent=(\S+).*mean_score_diff=([-0-9.]+)")


def cap_token(value):
    return f"{int(round(float(value) * 100)):03d}"


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
    for line in path.read_text(encoding="utf-8").splitlines():
        match = MEAN_RE.search(line)
        if match:
            means[match.group(1)] = float(match.group(2))
    return means


def selection_tuple(means, metric_opponents):
    metric = min(means[name] for name in metric_opponents)
    all_min = min(means.values())
    all_mean = sum(means.values()) / max(1, len(means))
    return (metric, all_min, all_mean)


def train_and_eval_cap(args, cap, index):
    token = cap_token(cap)
    checkpoint = CHECKPOINT_DIR / f"sweep_cap{token}_latest.pt"
    best_checkpoint = checkpoint.with_name(checkpoint.stem + "_best.pt")
    train_log = LOG_DIR / f"sweep_cap{token}_train.log"
    eval_log = LOG_DIR / f"sweep_cap{token}_eval.log"
    seed = args.seed + index * args.seed_stride

    train_command = [
        sys.executable,
        "train_target_selection_ppo_v3.py",
        "--seed",
        str(seed),
        "--resume-path",
        str(args.resume_path),
        "--iterations",
        str(args.sweep_iterations),
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
        "--cluster-signal-max-abs",
        str(cap),
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
        "train_target_selection_ppo_v3.py",
        "--eval-only",
        "--checkpoint-path",
        str(best_checkpoint),
        "--eval-opponents",
        *args.eval_opponents,
        "--eval-seeds",
        *[str(seed_value) for seed_value in args.sweep_eval_seeds],
        "--eval-steps",
        str(args.eval_steps),
    ]
    run_logged(eval_command, eval_log)
    means = parse_eval_log(eval_log)
    missing = [name for name in args.eval_opponents if name not in means]
    if missing:
        raise RuntimeError(f"Missing eval results for {missing} in {eval_log}")

    return {
        "cap": cap,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "train_log": str(train_log),
        "eval_log": str(eval_log),
        "means": means,
        "selection_tuple": selection_tuple(means, args.metric_opponents),
    }


def run_league(args, best_cap):
    league_log = LOG_DIR / "league_after_sweep_stdout.log"
    command = [
        sys.executable,
        "league_train_v3.py",
        "--generations",
        str(args.league_generations),
        "--iterations",
        str(args.league_iterations),
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
        "--cluster-signal-max-abs",
        str(best_cap),
        "--base-opponent-mix",
        args.opponent_mix,
        "--league-weight",
        str(args.league_weight),
        "--league-recency-decay",
        str(args.league_recency_decay),
        "--train-eval-seeds",
        *[str(seed_value) for seed_value in args.train_eval_seeds],
        "--gate-eval-seeds",
        *[str(seed_value) for seed_value in args.gate_eval_seeds],
        "--eval-steps",
        str(args.eval_steps),
    ]
    run_logged(command, league_log)
    return league_log


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caps", nargs="+", type=float, default=[0.20, 0.35, 0.50])
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--seed-stride", type=int, default=100)
    parser.add_argument("--resume-path", type=Path, default=DEFAULT_RESUME_PATH)
    parser.add_argument("--sweep-iterations", type=int, default=10)
    parser.add_argument("--league-generations", type=int, default=5)
    parser.add_argument("--league-iterations", type=int, default=20)
    parser.add_argument("--rollout-steps", type=int, default=3000)
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
        help="Primary sweep choice metric is the min score over these opponents.",
    )
    parser.add_argument("--train-eval-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--sweep-eval-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--gate-eval-seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--league-weight", type=float, default=0.35)
    parser.add_argument("--league-recency-decay", type=float, default=0.75)
    return parser.parse_args()


def main():
    args = parse_args()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for index, cap in enumerate(args.caps):
        result = train_and_eval_cap(args, cap, index)
        results.append(result)
        print(
            f"sweep cap={cap:.2f} score={result['selection_tuple']} "
            f"means={result['means']} train_log={result['train_log']} eval_log={result['eval_log']}",
            flush=True,
        )

    best = max(results, key=lambda row: tuple(row["selection_tuple"]))
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metric_opponents": args.metric_opponents,
        "best_cap": best["cap"],
        "best_result": best,
        "all_results": results,
        "league_iterations": args.league_iterations,
        "league_generations": args.league_generations,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"sweep_best cap={best['cap']:.2f} summary={SUMMARY_PATH}", flush=True)

    league_log = run_league(args, best["cap"])
    print(f"league_started_with_cap={best['cap']:.2f} log={league_log}", flush=True)


if __name__ == "__main__":
    main()
