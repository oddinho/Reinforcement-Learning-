import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PART2_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PART2_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PART2_ROOT) not in sys.path:
    sys.path.insert(0, str(PART2_ROOT))

from environments.collector.wrappers import CollectorGymEnv

from deterministic_agents import bfs_agent, rollout_gated_agent


ACTION_DIM = 4
REPORT_DIR = PART2_ROOT / "autoresearch_reports"


@dataclass(frozen=True)
class SweepVariant:
    name: str
    cluster_radius: int = 4
    cluster_weight: float = 0.75
    distance_weight: float = 1.0
    opponent_weight: float = 0.5
    route_weight: float = 8.0
    route_depth: int = 3
    route_decay: float = 0.7
    switch_margin: float = 1.5
    max_extra_distance: int = 3
    min_cluster_size: int = 2


VARIANTS = [
    SweepVariant(name="base"),
    SweepVariant(name="switch_2", switch_margin=2.0),
    SweepVariant(name="switch_25", switch_margin=2.5),
    SweepVariant(name="extra_2", max_extra_distance=2),
    SweepVariant(name="extra_2_switch_2", max_extra_distance=2, switch_margin=2.0),
    SweepVariant(name="route_5", route_weight=5.0),
    SweepVariant(name="route_5_switch_2", route_weight=5.0, switch_margin=2.0),
    SweepVariant(name="route_10", route_weight=10.0),
    SweepVariant(name="depth_2", route_depth=2),
    SweepVariant(name="depth_2_switch_2", route_depth=2, switch_margin=2.0),
    SweepVariant(name="radius_3", cluster_radius=3),
    SweepVariant(name="radius_3_extra_2_switch_2", cluster_radius=3, max_extra_distance=2, switch_margin=2.0),
]


def agent_config(variant, seed):
    values = {
        key: getattr(variant, key)
        for key in rollout_gated_agent.DEFAULT_CONFIG
    }
    values["seed"] = seed
    values["action_space"] = ACTION_DIM
    return SimpleNamespace(**values)


def run_match(variant, seed, max_steps):
    env = CollectorGymEnv(numpy_output=True)
    agent = rollout_gated_agent.Agent(agent_config(variant, seed))
    opponent = bfs_agent.Agent(SimpleNamespace(seed=seed, action_space=ACTION_DIM))
    agent.load()
    opponent.load()

    obs, info = env.reset(seed=seed)
    total_reward = np.zeros(2, dtype=np.float32)
    steps = 0

    for _ in range(max_steps):
        obs, reward, terminated, truncated, info = env.step(
            {
                "player_0": agent.act(obs["player_0"]),
                "player_1": opponent.act(obs["player_1"]),
            }
        )
        total_reward += np.asarray(reward, dtype=np.float32)
        steps += 1
        done = bool(np.asarray(terminated).any() or np.asarray(truncated).any())
        if done:
            break

    points = np.asarray(info["state"].team_points, dtype=int)
    env.close()
    return {
        "variant": variant.name,
        "opponent": "bfs",
        "seed": int(seed),
        "steps": int(steps),
        "team_points": points.tolist(),
        "score_diff": int(points[0] - points[1]),
        "reward_diff": float(total_reward[0] - total_reward[1]),
    }


def summarize(rows):
    by_variant = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)

    summary = []
    for variant, matches in sorted(by_variant.items()):
        score_diffs = [row["score_diff"] for row in matches]
        reward_diffs = [row["reward_diff"] for row in matches]
        loss_count = sum(diff < 0 for diff in score_diffs)
        tie_count = sum(diff == 0 for diff in score_diffs)
        win_count = sum(diff > 0 for diff in score_diffs)
        mean_score_diff = float(np.mean(score_diffs))
        summary.append(
            {
                "variant": variant,
                "matches": len(matches),
                "mean_score_diff": mean_score_diff,
                "median_score_diff": float(np.median(score_diffs)),
                "min_score_diff": int(np.min(score_diffs)),
                "max_score_diff": int(np.max(score_diffs)),
                "win_count": int(win_count),
                "tie_count": int(tie_count),
                "loss_count": int(loss_count),
                "mean_reward_diff": float(np.mean(reward_diffs)),
                "robust_score": float(mean_score_diff - 2.0 * loss_count),
            }
        )
    return sorted(summary, key=lambda row: (row["robust_score"], row["mean_score_diff"]), reverse=True)


def write_report(path, rows, summary, seeds, max_steps):
    variant_map = {variant.name: variant for variant in VARIANTS}
    lines = [
        "# Focused Rollout-Gated Sweep",
        "",
        "## Setup",
        "",
        f"- Seeds: `{list(seeds)}`",
        "- Opponent: `bfs`",
        f"- Max steps per match: `{max_steps}`",
        "- Objective: `robust_score = mean_score_diff - 2.0 * loss_count`",
        "",
        "## Summary",
        "",
        "| Variant | Matches | Mean | Median | Min | Max | W/T/L | Robust Score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| "
            f"{row['variant']} | {row['matches']} | "
            f"{row['mean_score_diff']:.2f} | {row['median_score_diff']:.2f} | "
            f"{row['min_score_diff']} | {row['max_score_diff']} | "
            f"{row['win_count']}/{row['tie_count']}/{row['loss_count']} | "
            f"{row['robust_score']:.2f} |"
        )

    lines.extend(["", "## Variant Configs", ""])
    for row in summary:
        variant = variant_map[row["variant"]]
        lines.append(f"### {variant.name}")
        lines.append("")
        lines.append(f"- `{json.dumps(variant.__dict__, sort_keys=True)}`")
        lines.append("")

    lines.extend(
        [
            "## Raw Matches",
            "",
            "| Variant | Seed | Score | Diff | Reward Diff |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row['variant']} | {row['seed']} | "
            f"{row['team_points'][0]}-{row['team_points'][1]} | "
            f"{row['score_diff']} | {row['reward_diff']:.2f} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORT_DIR / "rollout_gated_sweep_20seed.md",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=REPORT_DIR / "rollout_gated_sweep_20seed.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    total = len(VARIANTS) * len(args.seeds)
    completed = 0

    for variant in VARIANTS:
        for seed in args.seeds:
            row = run_match(variant, seed, args.max_steps)
            rows.append(row)
            completed += 1
            print(
                f"[{completed}/{total}] {variant.name} vs bfs seed={seed} "
                f"score={row['team_points'][0]}-{row['team_points'][1]} "
                f"diff={row['score_diff']}",
                flush=True,
            )

    summary = summarize(rows)
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(
        json.dumps(
            {
                "seeds": args.seeds,
                "opponent": "bfs",
                "max_steps": args.max_steps,
                "objective": "robust_score = mean_score_diff - 2.0 * loss_count",
                "variants": [variant.__dict__ for variant in VARIANTS],
                "summary": summary,
                "matches": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(args.report_path, rows, summary, args.seeds, args.max_steps)
    print(f"wrote_json={args.json_path}", flush=True)
    print(f"wrote_report={args.report_path}", flush=True)


if __name__ == "__main__":
    main()
