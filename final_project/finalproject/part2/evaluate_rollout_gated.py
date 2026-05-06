import argparse
import json
import sys
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

from agents.baseline.agent import Agent as BaselineAgent
from agents.random.agent import Agent as RandomAgent
from environments.collector.wrappers import CollectorGymEnv

from deterministic_agents import bfs_agent, rollout_gated_agent


ACTION_DIM = 4
REPORT_DIR = PART2_ROOT / "autoresearch_reports"


def build_agent(name, seed):
    if name == "bfs_reference":
        agent = bfs_agent.Agent(SimpleNamespace(seed=seed, action_space=ACTION_DIM))
    elif name == "rollout_gated":
        agent = rollout_gated_agent.Agent(
            SimpleNamespace(seed=seed, action_space=ACTION_DIM)
        )
    else:
        raise ValueError(f"Unknown agent: {name}")
    agent.load()
    return agent


def build_opponent(name, seed):
    if name == "baseline":
        agent = BaselineAgent(SimpleNamespace(epsilon=0.3, seed=seed, action_space=ACTION_DIM))
    elif name == "bfs":
        agent = bfs_agent.Agent(SimpleNamespace(seed=seed, action_space=ACTION_DIM))
    elif name == "random":
        agent = RandomAgent(SimpleNamespace(seed=seed, action_space=ACTION_DIM))
    else:
        raise ValueError(f"Unknown opponent: {name}")
    agent.load()
    return agent


def run_match(agent_name, opponent_name, seed, max_steps):
    env = CollectorGymEnv(numpy_output=True)
    agent = build_agent(agent_name, seed)
    opponent = build_opponent(opponent_name, seed)
    obs, info = env.reset(seed=seed)
    total_reward = np.zeros(2, dtype=np.float32)

    steps = 0
    for _ in range(max_steps):
        action0 = agent.act(obs["player_0"])
        action1 = opponent.act(obs["player_1"])
        obs, reward, terminated, truncated, info = env.step(
            {"player_0": action0, "player_1": action1}
        )
        total_reward += np.asarray(reward, dtype=np.float32)
        steps += 1
        done = bool(np.asarray(terminated).any() or np.asarray(truncated).any())
        if done:
            break

    points = np.asarray(info["state"].team_points, dtype=int)
    env.close()
    return {
        "agent": agent_name,
        "opponent": opponent_name,
        "seed": int(seed),
        "steps": int(steps),
        "team_points": points.tolist(),
        "score_diff": int(points[0] - points[1]),
        "reward_diff": float(total_reward[0] - total_reward[1]),
    }


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["agent"], row["opponent"]), []).append(row)

    summary = []
    for (agent, opponent), matches in sorted(groups.items()):
        score_diffs = [row["score_diff"] for row in matches]
        reward_diffs = [row["reward_diff"] for row in matches]
        summary.append(
            {
                "agent": agent,
                "opponent": opponent,
                "matches": len(matches),
                "mean_score_diff": float(np.mean(score_diffs)),
                "median_score_diff": float(np.median(score_diffs)),
                "min_score_diff": int(np.min(score_diffs)),
                "max_score_diff": int(np.max(score_diffs)),
                "loss_count": int(sum(diff < 0 for diff in score_diffs)),
                "tie_count": int(sum(diff == 0 for diff in score_diffs)),
                "win_count": int(sum(diff > 0 for diff in score_diffs)),
                "mean_reward_diff": float(np.mean(reward_diffs)),
            }
        )
    return summary


def write_report(path, rows, summary, seeds, opponents, max_steps):
    lines = [
        "# Rollout-Gated Evaluation",
        "",
        "## Setup",
        "",
        f"- Seeds: `{list(seeds)}`",
        f"- Opponents: `{list(opponents)}`",
        f"- Max steps per match: `{max_steps}`",
        "- Agents: `bfs_reference`, `rollout_gated`",
        "",
        "## Rollout-Gated Config",
        "",
        f"- `{json.dumps(rollout_gated_agent.DEFAULT_CONFIG, sort_keys=True)}`",
        "",
        "## Summary",
        "",
        "| Agent | Opponent | Matches | Mean Diff | Median | Min | Max | W/T/L | Mean Reward Diff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| "
            f"{row['agent']} | {row['opponent']} | {row['matches']} | "
            f"{row['mean_score_diff']:.2f} | {row['median_score_diff']:.2f} | "
            f"{row['min_score_diff']} | {row['max_score_diff']} | "
            f"{row['win_count']}/{row['tie_count']}/{row['loss_count']} | "
            f"{row['mean_reward_diff']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Raw Matches",
            "",
            "| Agent | Opponent | Seed | Score | Diff | Reward Diff |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row['agent']} | {row['opponent']} | {row['seed']} | "
            f"{row['team_points'][0]}-{row['team_points'][1]} | "
            f"{row['score_diff']} | {row['reward_diff']:.2f} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(50)))
    parser.add_argument("--opponents", nargs="+", choices=["baseline", "bfs", "random"], default=["baseline", "bfs"])
    parser.add_argument("--agents", nargs="+", choices=["bfs_reference", "rollout_gated"], default=["bfs_reference", "rollout_gated"])
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORT_DIR / "rollout_gated_eval_latest.md",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=REPORT_DIR / "rollout_gated_eval_latest.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    total = len(args.agents) * len(args.opponents) * len(args.seeds)
    completed = 0

    for agent_name in args.agents:
        for opponent_name in args.opponents:
            for seed in args.seeds:
                row = run_match(agent_name, opponent_name, seed, args.max_steps)
                rows.append(row)
                completed += 1
                print(
                    f"[{completed}/{total}] {agent_name} vs {opponent_name} "
                    f"seed={seed} score={row['team_points'][0]}-{row['team_points'][1]} "
                    f"diff={row['score_diff']}",
                    flush=True,
                )

    summary = summarize(rows)
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(
        json.dumps(
            {
                "seeds": args.seeds,
                "opponents": args.opponents,
                "agents": args.agents,
                "max_steps": args.max_steps,
                "rollout_gated_config": rollout_gated_agent.DEFAULT_CONFIG,
                "summary": summary,
                "matches": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(args.report_path, rows, summary, args.seeds, args.opponents, args.max_steps)
    print(f"wrote_json={args.json_path}", flush=True)
    print(f"wrote_report={args.report_path}", flush=True)


if __name__ == "__main__":
    main()
