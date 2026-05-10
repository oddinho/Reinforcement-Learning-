"""Record a replay from a trained target-selection PPO checkpoint."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


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
from train_target_selection_ppo import (
    DEFAULT_CHECKPOINT,
    load_checkpoint,
)
from target_selection_ppo_mlp import (
    ACTION_DIM,
    DeterministicTargetSelectionAgent,
    TargetSelectionMLP,
)


DEFAULT_OUTPUT = PART2_ROOT / "recordings" / "target_selection_ppo_vs_bfs_seed0.json"


def build_opponent(name, seed):
    if name == "bfs":
        agent = bfs_agent.Agent(SimpleNamespace(seed=seed, action_space=ACTION_DIM))
    elif name == "rollout_gated":
        agent = rollout_gated_agent.Agent(SimpleNamespace(seed=seed, action_space=ACTION_DIM))
    elif name == "baseline":
        agent = BaselineAgent(SimpleNamespace(epsilon=0.3, seed=seed, action_space=ACTION_DIM))
    elif name == "random":
        agent = RandomAgent(SimpleNamespace(seed=seed, action_space=ACTION_DIM))
    else:
        raise ValueError(f"Unknown opponent: {name}")
    agent.load()
    return agent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--opponent", default="bfs", choices=["bfs", "rollout_gated", "baseline", "random"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TargetSelectionMLP().to(device)
    loaded = load_checkpoint(model, args.checkpoint_path, device)
    if not loaded:
        raise FileNotFoundError(f"Could not load checkpoint: {args.checkpoint_path}")
    model.eval()

    env = RecordEpisode(
        CollectorGymEnv(numpy_output=True),
        save_on_reset=False,
        save_on_close=False,
    )
    agent = DeterministicTargetSelectionAgent(model, device, seed=args.seed)
    opponent = build_opponent(args.opponent, args.seed)

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
    print(f"team_points={points[0]}-{points[1]}")
    print(f"score_diff={int(points[0] - points[1])}")
    print(f"reward_diff={float(total_reward[0] - total_reward[1]):.1f}")
    print(f"replay={args.output}")


if __name__ == "__main__":
    main()
