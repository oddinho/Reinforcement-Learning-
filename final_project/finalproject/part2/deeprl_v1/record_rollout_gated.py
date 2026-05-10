import argparse
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
from environments.collector.wrappers import CollectorGymEnv, RecordEpisode

from deterministic_agents import bfs_agent, rollout_gated_agent


ACTION_DIM = 4
DEFAULT_OUTPUT = PART2_ROOT / "recordings" / "rollout_gated_route10_vs_bfs_seed0.json"


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", default="bfs", choices=["baseline", "bfs", "random"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cluster-radius", type=int, default=4)
    parser.add_argument("--cluster-weight", type=float, default=0.75)
    parser.add_argument("--distance-weight", type=float, default=1.0)
    parser.add_argument("--opponent-weight", type=float, default=0.5)
    parser.add_argument("--route-weight", type=float, default=10.0)
    parser.add_argument("--route-depth", type=int, default=3)
    parser.add_argument("--route-decay", type=float, default=0.7)
    parser.add_argument("--switch-margin", type=float, default=1.5)
    parser.add_argument("--max-extra-distance", type=int, default=3)
    parser.add_argument("--min-cluster-size", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    env = RecordEpisode(
        CollectorGymEnv(numpy_output=True),
        save_on_reset=False,
        save_on_close=False,
    )
    agent = rollout_gated_agent.Agent(
        SimpleNamespace(
            seed=args.seed,
            action_space=ACTION_DIM,
            cluster_radius=args.cluster_radius,
            cluster_weight=args.cluster_weight,
            distance_weight=args.distance_weight,
            opponent_weight=args.opponent_weight,
            route_weight=args.route_weight,
            route_depth=args.route_depth,
            route_decay=args.route_decay,
            switch_margin=args.switch_margin,
            max_extra_distance=args.max_extra_distance,
            min_cluster_size=args.min_cluster_size,
        )
    )
    opponent = build_opponent(args.opponent, args.seed)
    agent.load()

    obs, info = env.reset(seed=args.seed)
    total_reward = np.zeros(2, dtype=np.float32)

    steps = 0
    for _ in range(args.max_steps):
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

    env.save_episode(str(args.output))
    env.close()

    points = np.asarray(info["state"].team_points, dtype=int)
    print(f"steps={steps}")
    print(f"opponent={args.opponent}")
    print(f"seed={args.seed}")
    print(f"route_weight={args.route_weight}")
    print(f"team_points={points[0]}-{points[1]}")
    print(f"score_diff={int(points[0] - points[1])}")
    print(f"reward_diff={float(total_reward[0] - total_reward[1]):.1f}")
    print(f"replay={args.output}")


if __name__ == "__main__":
    main()
