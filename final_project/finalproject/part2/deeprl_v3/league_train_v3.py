"""Outer-loop frozen-snapshot self-play for deeprl_v3.

This script trains a candidate, evaluates its best checkpoint, accepts it only
if it improves enough, and then appends the accepted checkpoint to a frozen
league pool for the next generation.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = EXPERIMENT_ROOT / "checkpoints"
LOG_DIR = EXPERIMENT_ROOT / "training_logs"
POOL_PATH = EXPERIMENT_ROOT / "league_pool.json"
DEFAULT_RESUME_PATH = (
    EXPERIMENT_ROOT.parent
    / "checkpoints"
    / "target_selection_ppo_mlp_position_selfplay_latest_best.pt"
)

MEAN_RE = re.compile(r"eval opponent=(\S+).*mean_score_diff=([-0-9.]+)")
SEED_RE = re.compile(r"eval_seed seed=\d+.*score_diff=([-0-9]+)")


def parse_mix(text):
    result = []
    for part in text.split(","):
        name, weight = part.split(":")
        result.append((name.strip(), float(weight)))
    total = sum(weight for _name, weight in result)
    if total <= 0.0:
        raise ValueError("Opponent mix weights must sum to > 0.")
    return [(name, weight / total) for name, weight in result]


def format_mix(rows):
    return ",".join(f"{name}:{weight:.8f}" for name, weight in rows if weight > 0.0)


def mix_with_league(base_mix, league_weight, has_league):
    rows = parse_mix(base_mix)
    if not has_league or league_weight <= 0.0:
        return format_mix(rows)
    scaled = [(name, weight * (1.0 - league_weight)) for name, weight in rows]
    scaled.append(("league", league_weight))
    return format_mix(scaled)


def recency_weights(count, decay):
    """Newest accepted snapshot gets the largest weight, older ones decay."""

    if count <= 0:
        return []
    rows = [float(decay) ** age for age in reversed(range(count))]
    total = sum(rows)
    return [weight / total for weight in rows]


def load_pool(path=POOL_PATH):
    if not path.exists():
        return {"accepted": [], "best_score": None}
    return json.loads(path.read_text(encoding="utf-8"))


def save_pool(pool, path=POOL_PATH):
    path.write_text(json.dumps(pool, indent=2), encoding="utf-8")


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
    score_diffs = {}
    current_opponent = None
    for line in path.read_text(encoding="utf-8").splitlines():
        mean_match = MEAN_RE.search(line)
        if mean_match:
            current_opponent = mean_match.group(1)
            means[current_opponent] = float(mean_match.group(2))
            score_diffs.setdefault(current_opponent, [])
            continue
        seed_match = SEED_RE.search(line)
        if seed_match and current_opponent is not None:
            score_diffs.setdefault(current_opponent, []).append(int(seed_match.group(1)))
    wins = {
        opponent: sum(1 for score_diff in rows if score_diff > 0)
        for opponent, rows in score_diffs.items()
    }
    return means, wins, score_diffs


def checkpoint_iteration(path):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint.get("iteration", -1))


def train_generation(args, generation, league_checkpoints, resume_path):
    checkpoint = CHECKPOINT_DIR / f"league_gen{generation:02d}_latest.pt"
    train_log = LOG_DIR / f"league_gen{generation:02d}_train.log"
    opponent_mix = mix_with_league(
        args.base_opponent_mix,
        args.league_weight,
        bool(league_checkpoints),
    )
    selection_opponents = [
        opponent
        for opponent in args.selection_opponents
        if opponent != "league" or league_checkpoints
    ]
    selection_mean_opponents = [
        opponent
        for opponent in args.selection_mean_opponents
        if opponent != "league" or league_checkpoints
    ]
    eval_opponents = list(args.eval_opponents)
    for opponent in selection_opponents + selection_mean_opponents:
        if opponent not in eval_opponents:
            eval_opponents.append(opponent)

    command = [
        sys.executable,
        "train_target_selection_ppo_v3.py",
        "--seed",
        str(args.seed + generation * args.seed_stride),
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
        "--cluster-signal-max-abs",
        str(args.cluster_signal_max_abs),
        "--opponent-mix",
        opponent_mix,
        "--eval-opponents",
        *eval_opponents,
        "--selection-opponents",
        *selection_opponents,
        "--selection-mean-opponents",
        *selection_mean_opponents,
        "--selection-min-wins",
        str(args.selection_min_wins),
        "--selection-min-mean-score",
        str(args.selection_min_mean_score),
        "--eval-seeds",
        *[str(seed) for seed in args.train_eval_seeds],
        "--checkpoint-path",
        str(checkpoint),
    ]
    if resume_path is not None:
        command.extend(["--resume-path", str(resume_path)])
    if league_checkpoints:
        command.extend(["--league-checkpoints", *[str(path) for path in league_checkpoints]])
        command.extend(
            [
                "--league-checkpoint-weights",
                *[
                    f"{weight:.8f}"
                    for weight in recency_weights(
                        len(league_checkpoints),
                        args.league_recency_decay,
                    )
                ],
            ]
        )

    run_logged(command, train_log)
    return checkpoint.with_name(checkpoint.stem + "_best.pt"), train_log, opponent_mix


def evaluate_candidate(args, generation, checkpoint, league_checkpoints):
    eval_log = LOG_DIR / f"league_gen{generation:02d}_eval10.log"
    command = [
        sys.executable,
        "train_target_selection_ppo_v3.py",
        "--eval-only",
        "--checkpoint-path",
        str(checkpoint),
        "--eval-opponents",
        *args.gate_opponents,
        "--eval-seeds",
        *[str(seed) for seed in args.gate_eval_seeds],
        "--eval-steps",
        str(args.eval_steps),
    ]
    if "league" in args.gate_opponents and league_checkpoints:
        command.extend(["--league-checkpoints", *[str(path) for path in league_checkpoints]])

    run_logged(command, eval_log)
    means, wins, score_diffs = parse_eval_log(eval_log)
    missing = [name for name in args.gate_opponents if name not in means]
    if missing:
        raise RuntimeError(f"Missing eval results for {missing} in {eval_log}")
    missing_wins = [name for name in args.win_gate_opponents if name not in wins]
    if missing_wins:
        raise RuntimeError(f"Missing per-seed win results for {missing_wins} in {eval_log}")
    mean_gate_score = min(means[name] for name in args.mean_gate_opponents)
    win_gate_score = min(wins[name] for name in args.win_gate_opponents)
    return eval_log, means, wins, score_diffs, mean_gate_score, win_gate_score


def archive_accepted(generation, checkpoint):
    target = CHECKPOINT_DIR / f"ppo_league_gen{generation}.pt"
    shutil.copy2(checkpoint, target)
    return target


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument(
        "--start-generation",
        type=int,
        default=None,
        help=(
            "Optional explicit generation number for the first run. "
            "Use this to keep log names monotonic after manual/abandoned runs."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=1000,
        help="Added to the base seed each generation so rejected runs do not repeat.",
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rollout-steps", type=int, default=3000)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--terminal-win-bonus", type=float, default=5.0)
    parser.add_argument("--reward-mode", default="score_delta_cluster")
    parser.add_argument("--cluster-signal-max-abs", type=float, default=0.20)
    parser.add_argument(
        "--resume-path",
        type=Path,
        default=DEFAULT_RESUME_PATH,
        help=(
            "Initial checkpoint for generation 1. Later generations resume from "
            "the most recent accepted league checkpoint if one exists."
        ),
    )
    parser.add_argument(
        "--base-opponent-mix",
        default="bfs:0.15,rollout_gated:0.25,ppo:0.25,position_aware_ppo:0.35",
    )
    parser.add_argument(
        "--league-weight",
        type=float,
        default=0.35,
        help="Total probability assigned to accepted league snapshots after gen 1.",
    )
    parser.add_argument(
        "--league-recency-decay",
        type=float,
        default=0.75,
        help=(
            "Weight multiplier for each older accepted league snapshot. "
            "Newest gets largest weight; older snapshots stay present."
        ),
    )
    parser.add_argument(
        "--eval-opponents",
        nargs="+",
        default=["bfs", "rollout_gated", "ppo", "position_aware_ppo"],
    )
    parser.add_argument(
        "--selection-opponents",
        nargs="+",
        default=["position_aware_ppo", "league"],
    )
    parser.add_argument(
        "--selection-mean-opponents",
        nargs="+",
        default=["bfs", "rollout_gated", "ppo", "position_aware_ppo", "league"],
    )
    parser.add_argument("--selection-min-wins", type=int, default=3)
    parser.add_argument("--selection-min-mean-score", type=float, default=0.0)
    parser.add_argument(
        "--gate-opponents",
        nargs="+",
        default=["bfs", "rollout_gated", "ppo", "position_aware_ppo"],
        help="Opponents evaluated by the 10-seed accept/reject gate.",
    )
    parser.add_argument(
        "--mean-gate-opponents",
        nargs="+",
        default=["bfs", "rollout_gated", "ppo", "position_aware_ppo"],
        help="Opponents that must each have at least --mean-gate-min-score.",
    )
    parser.add_argument("--mean-gate-min-score", type=float, default=0.0)
    parser.add_argument(
        "--win-gate-opponents",
        nargs="+",
        default=["bfs", "rollout_gated", "ppo", "position_aware_ppo"],
        help="Opponents that must each be beaten on enough gate seeds.",
    )
    parser.add_argument("--win-gate-min-wins", type=int, default=6)
    parser.add_argument("--train-eval-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--gate-eval-seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument(
        "--initial-best-score",
        type=float,
        default=6.70,
        help="Current score to beat. Default is position-aware PPO rollout_gated 10-seed mean.",
    )
    parser.add_argument("--improve-margin", type=float, default=0.50)
    parser.add_argument("--accept-min-score", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    pool = load_pool()
    best_score = pool.get("best_score")
    if best_score is None:
        best_score = float(args.initial_best_score)
        pool["best_score"] = best_score
    accepted = pool.setdefault("accepted", [])
    league_checkpoints = [Path(row["path"]) for row in accepted]

    first_generation = (
        int(args.start_generation)
        if args.start_generation is not None
        else len(accepted) + 1
    )
    for generation in range(first_generation, first_generation + args.generations):
        resume_path = league_checkpoints[-1] if league_checkpoints else args.resume_path
        checkpoint, train_log, opponent_mix = train_generation(
            args,
            generation,
            league_checkpoints,
            resume_path,
        )
        (
            eval_log,
            means,
            wins,
            score_diffs,
            mean_gate_score,
            win_gate_score,
        ) = evaluate_candidate(
            args,
            generation,
            checkpoint,
            league_checkpoints,
        )
        accepted_candidate = (
            mean_gate_score >= args.mean_gate_min_score
            and win_gate_score >= args.win_gate_min_wins
        )

        print(
            f"generation={generation} iteration={checkpoint_iteration(checkpoint)} "
            f"mean_gate_score={mean_gate_score:.2f} "
            f"mean_gate_threshold={args.mean_gate_min_score:.2f} "
            f"win_gate_score={win_gate_score} "
            f"win_gate_threshold={args.win_gate_min_wins} "
            f"accepted={accepted_candidate} means={means} wins={wins} "
            f"train_log={train_log} eval_log={eval_log}",
            flush=True,
        )

        if accepted_candidate:
            archived = archive_accepted(generation, checkpoint)
            league_checkpoints.append(archived)
            pool["best_score"] = mean_gate_score
            accepted.append(
                {
                    "generation": generation,
                    "path": str(archived),
                    "source_checkpoint": str(checkpoint),
                    "iteration": checkpoint_iteration(checkpoint),
                    "mean_gate_score": mean_gate_score,
                    "win_gate_score": win_gate_score,
                    "means": means,
                    "wins": wins,
                    "score_diffs": score_diffs,
                    "opponent_mix": opponent_mix,
                    "train_log": str(train_log),
                    "eval_log": str(eval_log),
                    "accepted_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            save_pool(pool)

    save_pool(pool)


if __name__ == "__main__":
    main()
