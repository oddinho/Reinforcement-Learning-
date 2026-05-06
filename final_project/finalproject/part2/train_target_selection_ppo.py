#imports 

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


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
from target_selection_ppo_mlp import (
    ACTION_DIM,
    FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    DeterministicTargetSelectionAgent,
    TargetSelectionMLP,
    build_target_candidates,
    get_state_parts,
    safe_random_action,
    select_target_action,
)


CHECKPOINT_DIR = PART2_ROOT / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "target_selection_ppo_mlp_abandon_latest.pt"


@dataclass
class Transition:
    features: np.ndarray
    global_features: np.ndarray
    action_index: int
    log_prob: float
    value: float
    reward: float
    done: bool


def parse_opponent_mix(text):
    result = []
    for part in text.split(","):
        name, weight = part.split(":")
        name = name.strip()
        if name not in {"bfs", "rollout_gated", "baseline", "random"}:
            raise argparse.ArgumentTypeError(f"Unknown opponent: {name}")
        result.append((name, float(weight)))
    total = sum(weight for _name, weight in result)
    if total <= 0:
        raise argparse.ArgumentTypeError("Opponent weights must sum to > 0.")
    return tuple((name, weight / total) for name, weight in result)


def choose_opponent_name(rng, opponent_mix):
    names = [name for name, _weight in opponent_mix]
    probabilities = [weight for _name, weight in opponent_mix]
    return str(rng.choice(names, p=probabilities))


def build_opponent(name, seed):
    if name == "bfs":
        agent = bfs_agent.Agent(SimpleNamespace(seed=seed, action_space=ACTION_DIM))
    elif name == "rollout_gated":
        agent = rollout_gated_agent.Agent(
            SimpleNamespace(seed=seed, action_space=ACTION_DIM)
        )
    elif name == "baseline":
        agent = BaselineAgent(
            SimpleNamespace(epsilon=0.3, seed=seed, action_space=ACTION_DIM)
        )
    elif name == "random":
        agent = RandomAgent(SimpleNamespace(seed=seed, action_space=ACTION_DIM))
    else:
        raise ValueError(f"Unknown opponent: {name}")
    agent.load()
    return agent


def reward_diff(raw_reward):
    reward = np.asarray(raw_reward, dtype=np.float32)
    return float(reward[0] - reward[1])


def score_delta_diff(previous_info, next_info):
    previous_points = np.asarray(previous_info["state"].team_points, dtype=np.float32)
    next_points = np.asarray(next_info["state"].team_points, dtype=np.float32)
    previous_diff = float(previous_points[0] - previous_points[1])
    next_diff = float(next_points[0] - next_points[1])
    return next_diff - previous_diff


def training_reward(raw_reward, previous_info, next_info, reward_mode):
    if reward_mode == "env_diff":
        return reward_diff(raw_reward)
    if reward_mode == "score_delta":
        return score_delta_diff(previous_info, next_info)
    raise ValueError(f"Unknown reward mode: {reward_mode}")


def collect_rollout(model, args, device, rng, iteration):
    env = CollectorGymEnv(numpy_output=True)
    obs = None
    info = None
    opponent = None
    episode_seed = args.seed + iteration * 10_000
    transitions = []
    episode_returns = []
    episode_score_diffs = []
    current_return = 0.0
    rollout_reward = 0.0

    for _step in range(args.rollout_steps):
        if obs is None:
            opponent_name = choose_opponent_name(rng, args.opponent_mix)
            opponent = build_opponent(opponent_name, episode_seed)
            obs, info = env.reset(seed=episode_seed)
            episode_seed += 1
            current_return = 0.0

        candidates = build_target_candidates(obs["player_0"])
        if candidates is None:
            tile_map, self_pos, _opponent_pos, _items = get_state_parts(obs["player_0"])
            action = safe_random_action(tile_map, self_pos, rng)
            log_prob = 0.0
            value = 0.0
            action_index = 0
            store_transition = False
        else:
            action_index, action, log_prob, value = select_target_action(
                model=model,
                candidates=candidates,
                device=device,
                sample=True,
            )
            store_transition = True

        next_obs, raw_reward, terminated, truncated, next_info = env.step(
            {
                "player_0": action,
                "player_1": opponent.act(obs["player_1"]),
            }
        )
        reward = training_reward(raw_reward, info, next_info, args.reward_mode)
        done = bool(np.asarray(terminated).any() or np.asarray(truncated).any())
        current_return += reward
        rollout_reward += reward

        if store_transition:
            transitions.append(
                Transition(
                    features=candidates.features.astype(np.float32),
                    global_features=candidates.global_features.astype(np.float32),
                    action_index=int(action_index),
                    log_prob=float(log_prob),
                    value=float(value),
                    reward=float(reward),
                    done=done,
                )
            )

        if done:
            points = np.asarray(next_info["state"].team_points, dtype=int)
            episode_returns.append(current_return)
            episode_score_diffs.append(int(points[0] - points[1]))
            obs = None
            info = None
        else:
            obs = next_obs
            info = next_info

    env.close()
    return transitions, {
        "episodes": len(episode_returns),
        "mean_step_reward": float(rollout_reward / max(1, len(transitions))),
        "mean_return": float(np.mean(episode_returns)) if episode_returns else float("nan"),
        "mean_score_diff": float(np.mean(episode_score_diffs)) if episode_score_diffs else float("nan"),
    }


def compute_gae(transitions, gamma, gae_lambda):
    rewards = np.asarray([transition.reward for transition in transitions], dtype=np.float32)
    values = np.asarray(
        [transition.value for transition in transitions] + [0.0],
        dtype=np.float32,
    )
    dones = np.asarray([transition.done for transition in transitions], dtype=np.float32)

    advantages = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        gae = delta + gamma * gae_lambda * not_done * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
    return advantages.astype(np.float32), returns.astype(np.float32)


def make_batch(transitions, indices, advantages, returns, device):
    batch = [transitions[idx] for idx in indices]
    batch_size = len(batch)
    max_candidates = max(transition.features.shape[0] for transition in batch)

    features = np.zeros((batch_size, max_candidates, FEATURE_DIM), dtype=np.float32)
    global_features = np.zeros((batch_size, GLOBAL_FEATURE_DIM), dtype=np.float32)
    valid_mask = np.zeros((batch_size, max_candidates), dtype=bool)
    action_indices = np.zeros(batch_size, dtype=np.int64)
    old_log_probs = np.zeros(batch_size, dtype=np.float32)
    batch_advantages = np.zeros(batch_size, dtype=np.float32)
    batch_returns = np.zeros(batch_size, dtype=np.float32)

    for row, transition in enumerate(batch):
        n = transition.features.shape[0]
        features[row, :n] = transition.features
        global_features[row] = transition.global_features
        valid_mask[row, :n] = True
        action_indices[row] = transition.action_index
        old_log_probs[row] = transition.log_prob
        batch_advantages[row] = advantages[indices[row]]
        batch_returns[row] = returns[indices[row]]

    return (
        torch.from_numpy(features).to(device),
        torch.from_numpy(global_features).to(device),
        torch.from_numpy(valid_mask).to(device),
        torch.from_numpy(action_indices).to(device),
        torch.from_numpy(old_log_probs).to(device),
        torch.from_numpy(batch_advantages).to(device),
        torch.from_numpy(batch_returns).to(device),
    )


def ppo_update(model, optimizer, transitions, advantages, returns, args, device, rng):
    indices = list(range(len(transitions)))
    metrics = []
    model.train()

    for _epoch in range(args.ppo_epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start:start + args.batch_size]
            (
                features,
                global_features,
                valid_mask,
                action_indices,
                old_log_probs,
                batch_advantages,
                batch_returns,
            ) = make_batch(transitions, batch_indices, advantages, returns, device)

            logits, values = model(features, valid_mask, global_features)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(action_indices)
            entropy = dist.entropy().mean()

            ratios = torch.exp(log_probs - old_log_probs)
            unclipped = ratios * batch_advantages
            clipped = (
                torch.clamp(ratios, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                * batch_advantages
            )
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, batch_returns)
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (old_log_probs - log_probs).mean().abs()
            metrics.append(
                {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                    "approx_kl": float(approx_kl.item()),
                }
            )

    return {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]}


def live_evaluate(model, device, opponent_name, seed, max_steps):
    env = CollectorGymEnv(numpy_output=True)
    agent = DeterministicTargetSelectionAgent(model, device, seed=seed)
    opponent = build_opponent(opponent_name, seed)
    obs, info = env.reset(seed=seed)
    total_reward = np.zeros(2, dtype=np.float32)

    for step in range(1, max_steps + 1):
        obs, reward, terminated, truncated, info = env.step(
            {
                "player_0": agent.act(obs["player_0"]),
                "player_1": opponent.act(obs["player_1"]),
            }
        )
        total_reward += np.asarray(reward, dtype=np.float32)
        done = bool(np.asarray(terminated).any() or np.asarray(truncated).any())
        if done:
            break

    env.close()
    points = np.asarray(info["state"].team_points, dtype=int)
    return {
        "seed": int(seed),
        "steps": int(step),
        "team_points": [int(points[0]), int(points[1])],
        "score_diff": int(points[0] - points[1]),
        "reward_diff": float(total_reward[0] - total_reward[1]),
    }


def evaluate_many(model, device, opponent_name, seeds, max_steps):
    rows = [
        live_evaluate(model, device, opponent_name, seed, max_steps)
        for seed in seeds
    ]
    score_diffs = [row["score_diff"] for row in rows]
    reward_diffs = [row["reward_diff"] for row in rows]
    print(
        f"eval opponent={opponent_name} seeds={list(seeds)} "
        f"mean_score_diff={float(np.mean(score_diffs)):.2f} "
        f"min={int(np.min(score_diffs))} max={int(np.max(score_diffs))} "
        f"mean_reward_diff={float(np.mean(reward_diffs)):.2f}",
        flush=True,
    )
    for row in rows:
        print(
            f"eval_seed seed={row['seed']} points={row['team_points'][0]}-{row['team_points'][1]} "
            f"score_diff={row['score_diff']} reward_diff={row['reward_diff']:.1f}",
            flush=True,
        )
    return {
        "rows": rows,
        "mean_score_diff": float(np.mean(score_diffs)),
        "min_score_diff": int(np.min(score_diffs)),
        "max_score_diff": int(np.max(score_diffs)),
        "mean_reward_diff": float(np.mean(reward_diffs)),
    }


def save_checkpoint(model, path, iteration, metrics, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "iteration": int(iteration),
            "metrics": metrics,
            "args": vars(args),
            "feature_dim": FEATURE_DIM,
            "global_feature_dim": GLOBAL_FEATURE_DIM,
            "hidden_dim": int(args.hidden_dim),
            "architecture": "target_selection_mlp_tanh_2x_abandon_rich_critic",
        },
        path,
    )


def load_checkpoint(model, path, device):
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return False
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--best-checkpoint-path", type=Path, default=None)
    parser.add_argument("--resume-path", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rollout-steps", type=int, default=3000)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--reward-mode",
        choices=["score_delta", "env_diff"],
        default="score_delta",
        help="score_delta optimizes item-score changes; env_diff uses raw environment reward difference.",
    )
    parser.add_argument(
        "--opponent-mix",
        type=parse_opponent_mix,
        default=parse_opponent_mix("bfs:0.4,rollout_gated:0.6"),
    )
    parser.add_argument(
        "--eval-opponents",
        nargs="+",
        choices=["bfs", "rollout_gated", "baseline", "random"],
        default=["bfs", "rollout_gated"],
    )
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_checkpoint_path = args.best_checkpoint_path
    if best_checkpoint_path is None:
        best_checkpoint_path = args.checkpoint_path.with_name(
            args.checkpoint_path.stem + "_best.pt"
        )

    model = TargetSelectionMLP(hidden_dim=args.hidden_dim).to(device)
    load_path = args.resume_path
    if args.eval_only and load_path is None:
        load_path = args.checkpoint_path
    resumed = False
    if load_path is not None:
        resumed = load_checkpoint(model, load_path, device)
        if args.eval_only and not resumed:
            raise FileNotFoundError(f"Could not load checkpoint: {load_path}")

    if args.eval_only:
        print(
            f"target_selection_mlp eval_only checkpoint={load_path} device={device}",
            flush=True,
        )
        for opponent in args.eval_opponents:
            evaluate_many(model, device, opponent, args.eval_seeds, args.eval_steps)
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1.0e-5)
    print(
        f"target_selection_mlp device={device} resumed={resumed} "
        f"iterations={args.iterations} rollout_steps={args.rollout_steps} "
        f"hidden_dim={args.hidden_dim} reward_mode={args.reward_mode} "
        f"opponent_mix={args.opponent_mix}",
        flush=True,
    )

    best_eval_score = -float("inf")
    metrics = {}
    for iteration in range(1, args.iterations + 1):
        transitions, rollout_metrics = collect_rollout(model, args, device, rng, iteration)
        if not transitions:
            raise RuntimeError("No PPO transitions collected.")

        advantages, returns = compute_gae(transitions, args.gamma, args.gae_lambda)
        update_metrics = ppo_update(
            model,
            optimizer,
            transitions,
            advantages,
            returns,
            args,
            device,
            py_rng,
        )
        metrics = {**rollout_metrics, **update_metrics}
        print(
            f"ppo iter={iteration} transitions={len(transitions)} "
            f"episodes={rollout_metrics['episodes']} "
            f"mean_step_reward={rollout_metrics['mean_step_reward']:.3f} "
            f"rollout_return={rollout_metrics['mean_return']:.2f} "
            f"rollout_score_diff={rollout_metrics['mean_score_diff']:.2f} "
            f"loss={update_metrics['loss']:.4f} "
            f"policy={update_metrics['policy_loss']:.4f} "
            f"value={update_metrics['value_loss']:.4f} "
            f"entropy={update_metrics['entropy']:.4f} "
            f"kl={update_metrics['approx_kl']:.4f}",
            flush=True,
        )

        if iteration % args.eval_interval == 0:
            eval_metrics = {}
            for opponent in args.eval_opponents:
                opponent_metrics = evaluate_many(
                    model,
                    device,
                    opponent,
                    args.eval_seeds,
                    args.eval_steps,
                )
                eval_metrics[f"eval_{opponent}_mean_score_diff"] = opponent_metrics[
                    "mean_score_diff"
                ]
                eval_metrics[f"eval_{opponent}_min_score_diff"] = opponent_metrics[
                    "min_score_diff"
                ]
            if eval_metrics:
                eval_score = min(
                    value
                    for key, value in eval_metrics.items()
                    if key.endswith("_mean_score_diff")
                )
                metrics = {
                    **metrics,
                    **eval_metrics,
                    "eval_selection_score": float(eval_score),
                }
                if eval_score > best_eval_score:
                    best_eval_score = float(eval_score)
                    save_checkpoint(
                        model,
                        best_checkpoint_path,
                        iteration,
                        metrics,
                        args,
                    )
                    print(
                        f"saved_best_checkpoint={best_checkpoint_path} "
                        f"iteration={iteration} selection_score={best_eval_score:.2f}",
                        flush=True,
                    )
            save_checkpoint(model, args.checkpoint_path, iteration, metrics, args)

    save_checkpoint(model, args.checkpoint_path, args.iterations, metrics, args)
    print(f"saved_checkpoint={args.checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
