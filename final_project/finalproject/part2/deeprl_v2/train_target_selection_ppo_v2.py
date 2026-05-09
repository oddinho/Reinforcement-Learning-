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


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PART2_ROOT = EXPERIMENT_ROOT.parent
PROJECT_ROOT = PART2_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PART2_ROOT) not in sys.path:
    sys.path.insert(0, str(PART2_ROOT))
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from agents.baseline.agent import Agent as BaselineAgent
from agents.random.agent import Agent as RandomAgent
from environments.collector.wrappers import CollectorGymEnv

from deterministic_agents import bfs_agent, rollout_gated_agent
from target_selection_ppo_mlp_v2 import (
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


CHECKPOINT_DIR = EXPERIMENT_ROOT / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "target_selection_ppo_mlp_128_latest.pt"
DEFAULT_PPO_CHECKPOINT = (
    PART2_ROOT / "checkpoints" / "target_selection_ppo_mlp_abandon_30_latest_best.pt"
)
DEFAULT_POSITION_AWARE_PPO_CHECKPOINT = (
    PART2_ROOT
    / "checkpoints"
    / "target_selection_ppo_mlp_position_selfplay_latest_best.pt"
)
OPPONENT_CHOICES = (
    "bfs",
    "rollout_gated",
    "ppo",
    "position_aware_ppo",
    "league",
    "baseline",
    "random",
)
REAL_EVAL_OPPONENTS = ("bfs", "rollout_gated", "ppo", "position_aware_ppo")

FEATURE_CLUSTER3 = 10
FEATURE_CLUSTER5 = 11
FEATURE_ROUTE_VALUE = 12
FEATURE_CONTESTED = 14
FEATURE_OWN_FAVORED = 15
FEATURE_LOST_RACE = 16
FEATURE_ROUTE_IF_OWN_FAVORED = 17
FEATURE_ROUTE_IF_LOST = 18
FEATURE_BEHIND = 20
FEATURE_CENTER_CLUSTER_VALUE = 25


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
        if name not in OPPONENT_CHOICES:
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


def checkpoint_hidden_dim(path, device, default=64):
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        return int(checkpoint.get("hidden_dim", default))
    return int(default)


def build_ppo_checkpoint_opponent(path, seed, device):
    hidden_dim = checkpoint_hidden_dim(path, device)
    snapshot_model = TargetSelectionMLP(hidden_dim=hidden_dim).to(device)
    loaded = load_checkpoint(snapshot_model, path, device)
    if not loaded:
        raise FileNotFoundError(f"Could not load PPO opponent checkpoint: {path}")
    snapshot_model.eval()
    return DeterministicTargetSelectionAgent(
        snapshot_model,
        device,
        seed=seed,
    )


def choose_league_checkpoint(paths, seed):
    if not paths:
        raise ValueError("league opponent requires --league-checkpoints")
    index = abs(int(seed)) % len(paths)
    return Path(paths[index])


def build_opponent(name, seed, args=None, device=None):
    snapshot_device = device if device is not None else torch.device("cpu")
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
    elif name == "ppo":
        if args is None or args.ppo_checkpoint is None:
            raise ValueError("ppo opponent requires --ppo-checkpoint")
        agent = build_ppo_checkpoint_opponent(
            args.ppo_checkpoint,
            seed,
            snapshot_device,
        )
    elif name == "position_aware_ppo":
        if args is None or args.position_aware_ppo_checkpoint is None:
            raise ValueError(
                "position_aware_ppo opponent requires --position-aware-ppo-checkpoint"
            )
        agent = build_ppo_checkpoint_opponent(
            args.position_aware_ppo_checkpoint,
            seed,
            snapshot_device,
        )
    elif name == "league":
        if args is None:
            raise ValueError("league opponent requires args")
        checkpoint = choose_league_checkpoint(args.league_checkpoints, seed)
        agent = build_ppo_checkpoint_opponent(checkpoint, seed, snapshot_device)
    else:
        raise ValueError(f"Unknown opponent: {name}")
    if hasattr(agent, "load"):
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
    if reward_mode in {"score_delta", "score_delta_shaped", "score_delta_cluster"}:
        return score_delta_diff(previous_info, next_info)
    raise ValueError(f"Unknown reward mode: {reward_mode}")


def point_deltas(previous_info, next_info):
    previous_points = np.asarray(previous_info["state"].team_points, dtype=np.float32)
    next_points = np.asarray(next_info["state"].team_points, dtype=np.float32)
    delta = next_points - previous_points
    return float(delta[0]), float(delta[1])


def target_shaping_reward(candidates, action_index, previous_info, next_info, args):
    """Dense target-selection hints for cluster/race behavior.

    The main training reward is still score-difference delta. These small terms
    make the immediate target-selection consequences less sparse:

    - reward own-favored cluster/route choices
    - reward front-running a contested item when we collect
    - penalize chasing targets the opponent is clearly favored to reach first
    """

    if candidates is None:
        return 0.0
    if action_index < 0 or action_index >= candidates.features.shape[0]:
        return 0.0

    row = candidates.features[action_index]
    cluster_value = 0.5 * (float(row[FEATURE_CLUSTER3]) + float(row[FEATURE_CLUSTER5]))
    route_value = float(row[FEATURE_ROUTE_VALUE])
    own_favored = float(row[FEATURE_OWN_FAVORED])
    lost_race = float(row[FEATURE_LOST_RACE])
    route_if_own_favored = float(row[FEATURE_ROUTE_IF_OWN_FAVORED])
    route_if_lost = float(row[FEATURE_ROUTE_IF_LOST])
    contested = float(row[FEATURE_CONTESTED])
    behind = float(row[FEATURE_BEHIND])

    cluster_bonus = args.shaping_cluster_bonus * own_favored * cluster_value
    route_bonus = args.shaping_route_bonus * route_if_own_favored

    own_delta, _opponent_delta = point_deltas(previous_info, next_info)
    front_run_bonus = 0.0
    if own_delta > 0.0:
        front_run_pressure = max(contested, own_favored * cluster_value)
        front_run_bonus = args.shaping_front_run_bonus * own_delta * front_run_pressure

    lost_penalty = args.shaping_lost_race_penalty * lost_race * (1.0 + behind)
    lost_route_penalty = args.shaping_lost_route_penalty * route_if_lost

    shaping = (
        cluster_bonus
        + route_bonus
        + front_run_bonus
        - lost_penalty
        - lost_route_penalty
    )
    return float(
        np.clip(
            shaping,
            -args.shaping_max_abs,
            args.shaping_max_abs,
        )
    )


def cluster_signal_reward(candidates, action_index, args):
    """Focused dense signal for choosing own-favored cluster/route targets."""

    if candidates is None:
        return 0.0
    if action_index < 0 or action_index >= candidates.features.shape[0]:
        return 0.0

    row = candidates.features[action_index]
    cluster_value = 0.5 * (float(row[FEATURE_CLUSTER3]) + float(row[FEATURE_CLUSTER5]))
    route_value = float(row[FEATURE_ROUTE_VALUE])
    own_favored = float(row[FEATURE_OWN_FAVORED])
    lost_race = float(row[FEATURE_LOST_RACE])
    center_cluster_value = float(row[FEATURE_CENTER_CLUSTER_VALUE])

    cluster_bonus = (
        args.cluster_signal_center_bonus
        * own_favored
        * center_cluster_value
    )
    route_bonus = args.cluster_signal_route_bonus * own_favored * route_value
    lost_cluster_penalty = (
        args.cluster_signal_lost_penalty
        * lost_race
        * cluster_value
    )

    signal = cluster_bonus + route_bonus - lost_cluster_penalty
    return float(
        np.clip(
            signal,
            -args.cluster_signal_max_abs,
            args.cluster_signal_max_abs,
        )
    )


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
    rollout_shaping_reward = 0.0
    rollout_terminal_bonus = 0.0

    for _step in range(args.rollout_steps):
        if obs is None:
            opponent_name = choose_opponent_name(rng, args.opponent_mix)
            opponent = build_opponent(opponent_name, episode_seed, args, device)
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
        shaping_reward = 0.0
        if store_transition:
            if args.reward_mode == "score_delta_shaped":
                shaping_reward = target_shaping_reward(
                    candidates,
                    action_index,
                    info,
                    next_info,
                    args,
                )
            elif args.reward_mode == "score_delta_cluster":
                shaping_reward = cluster_signal_reward(candidates, action_index, args)
            reward += shaping_reward
        done = bool(np.asarray(terminated).any() or np.asarray(truncated).any())
        terminal_bonus = 0.0
        if done and args.terminal_win_bonus != 0.0:
            points = np.asarray(next_info["state"].team_points, dtype=int)
            if int(points[0] - points[1]) > 0:
                terminal_bonus = float(args.terminal_win_bonus)
                reward += terminal_bonus
        current_return += reward
        rollout_reward += reward
        rollout_shaping_reward += shaping_reward
        rollout_terminal_bonus += terminal_bonus

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
        "mean_shaping_reward": float(rollout_shaping_reward / max(1, len(transitions))),
        "mean_terminal_bonus": float(rollout_terminal_bonus / max(1, len(transitions))),
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


def live_evaluate(model, device, opponent_name, seed, max_steps, args=None):
    env = CollectorGymEnv(numpy_output=True)
    agent = DeterministicTargetSelectionAgent(model, device, seed=seed)
    opponent = build_opponent(opponent_name, seed, args, device)
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


def evaluate_many(model, device, opponent_name, seeds, max_steps, args=None):
    rows = [
        live_evaluate(model, device, opponent_name, seed, max_steps, args)
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
            "architecture": "target_selection_mlp_tanh_2x128_position_v2",
        },
        path,
    )


def infer_checkpoint_feature_dim(checkpoint, state_dict):
    if isinstance(checkpoint, dict) and "feature_dim" in checkpoint:
        return int(checkpoint["feature_dim"])
    actor_weight = state_dict.get("actor.0.weight")
    if actor_weight is not None and actor_weight.ndim == 2:
        return int(actor_weight.shape[1])
    return FEATURE_DIM


def infer_checkpoint_global_feature_dim(checkpoint, old_feature_dim, state_dict):
    if isinstance(checkpoint, dict) and "global_feature_dim" in checkpoint:
        return int(checkpoint["global_feature_dim"])
    critic_weight = state_dict.get("critic.0.weight")
    if critic_weight is not None and critic_weight.ndim == 2:
        inferred = int(critic_weight.shape[1]) - old_feature_dim * 3 - 1
        if inferred > 0:
            return inferred
    return GLOBAL_FEATURE_DIM


def adapt_actor_input_weight(new_weight, old_weight):
    adapted = torch.zeros_like(new_weight)
    copy_cols = min(int(old_weight.shape[1]), int(new_weight.shape[1]))
    adapted[:, :copy_cols] = old_weight[:, :copy_cols]
    return adapted


def adapt_critic_input_weight(
    new_weight,
    old_weight,
    old_feature_dim,
    old_global_feature_dim,
):
    adapted = torch.zeros_like(new_weight)
    new_feature_dim = FEATURE_DIM
    copy_feature_dim = min(old_feature_dim, new_feature_dim)

    for block in range(3):
        old_start = block * old_feature_dim
        new_start = block * new_feature_dim
        adapted[:, new_start:new_start + copy_feature_dim] = old_weight[
            :, old_start:old_start + copy_feature_dim
        ]

    old_global_start = old_feature_dim * 3
    new_global_start = new_feature_dim * 3
    copy_global_dim = min(old_global_feature_dim, GLOBAL_FEATURE_DIM)
    adapted[:, new_global_start:new_global_start + copy_global_dim] = old_weight[
        :, old_global_start:old_global_start + copy_global_dim
    ]

    old_count_index = old_global_start + old_global_feature_dim
    new_count_index = new_global_start + GLOBAL_FEATURE_DIM
    if old_count_index < old_weight.shape[1] and new_count_index < new_weight.shape[1]:
        adapted[:, new_count_index:new_count_index + 1] = old_weight[
            :, old_count_index:old_count_index + 1
        ]
    return adapted


def adapt_checkpoint_state_dict(model, checkpoint, state_dict):
    """Load old 23-feature checkpoints into the widened 27-feature model.

    Existing feature columns keep their learned weights. Newly appended
    position-feature columns start at zero weight, so the run starts exactly
    from the old policy before learning how to use the new inputs.
    """

    model_state = model.state_dict()
    old_feature_dim = infer_checkpoint_feature_dim(checkpoint, state_dict)
    old_global_feature_dim = infer_checkpoint_global_feature_dim(
        checkpoint,
        old_feature_dim,
        state_dict,
    )
    adapted = {}

    for key, new_value in model_state.items():
        old_value = state_dict.get(key)
        if old_value is None:
            adapted[key] = new_value
            continue
        if old_value.shape == new_value.shape:
            adapted[key] = old_value
            continue
        if (
            key == "actor.0.weight"
            and old_value.ndim == 2
            and new_value.ndim == 2
            and old_value.shape[0] == new_value.shape[0]
        ):
            adapted[key] = adapt_actor_input_weight(new_value, old_value)
            continue
        if (
            key == "critic.0.weight"
            and old_value.ndim == 2
            and new_value.ndim == 2
            and old_value.shape[0] == new_value.shape[0]
        ):
            adapted[key] = adapt_critic_input_weight(
                new_value,
                old_value,
                old_feature_dim,
                old_global_feature_dim,
            )
            continue
        adapted[key] = new_value

    return adapted


def load_checkpoint(model, path, device):
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return False
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = adapt_checkpoint_state_dict(model, checkpoint, state_dict)
    model.load_state_dict(state_dict)
    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--best-checkpoint-path", type=Path, default=None)
    parser.add_argument("--resume-path", type=Path, default=None)
    parser.add_argument("--ppo-checkpoint", type=Path, default=DEFAULT_PPO_CHECKPOINT)
    parser.add_argument(
        "--position-aware-ppo-checkpoint",
        type=Path,
        default=DEFAULT_POSITION_AWARE_PPO_CHECKPOINT,
    )
    parser.add_argument(
        "--league-checkpoints",
        nargs="*",
        type=Path,
        default=[],
        help="Frozen accepted PPO snapshots sampled when opponent name is 'league'.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rollout-steps", type=int, default=6000)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--reward-mode",
        choices=["score_delta", "score_delta_shaped", "score_delta_cluster", "env_diff"],
        default="score_delta_cluster",
        help=(
            "score_delta optimizes item-score changes; score_delta_cluster adds "
            "focused own-favored cluster/route shaping; score_delta_shaped adds "
            "the older broad shaping; env_diff uses raw environment reward "
            "difference."
        ),
    )
    parser.add_argument("--shaping-cluster-bonus", type=float, default=0.02)
    parser.add_argument("--shaping-route-bonus", type=float, default=0.03)
    parser.add_argument("--shaping-front-run-bonus", type=float, default=0.05)
    parser.add_argument("--shaping-lost-race-penalty", type=float, default=0.04)
    parser.add_argument("--shaping-lost-route-penalty", type=float, default=0.03)
    parser.add_argument("--shaping-max-abs", type=float, default=0.15)
    parser.add_argument("--cluster-signal-center-bonus", type=float, default=0.08)
    parser.add_argument("--cluster-signal-route-bonus", type=float, default=0.06)
    parser.add_argument("--cluster-signal-lost-penalty", type=float, default=0.08)
    parser.add_argument("--cluster-signal-max-abs", type=float, default=0.20)
    parser.add_argument(
        "--terminal-win-bonus",
        type=float,
        default=5.0,
        help="Added only on the terminal transition when final score diff is positive.",
    )
    parser.add_argument(
        "--opponent-mix",
        type=parse_opponent_mix,
        default=parse_opponent_mix(
            "bfs:0.15,rollout_gated:0.25,ppo:0.25,position_aware_ppo:0.35"
        ),
    )
    parser.add_argument(
        "--eval-opponents",
        nargs="+",
        choices=OPPONENT_CHOICES,
        default=["bfs", "rollout_gated", "ppo", "position_aware_ppo"],
    )
    parser.add_argument(
        "--selection-opponents",
        nargs="+",
        choices=OPPONENT_CHOICES,
        default=list(REAL_EVAL_OPPONENTS),
        help=(
            "Opponents used for best-checkpoint selection. "
            "Use this to exclude auxiliary training opponents such as ppo_snapshot."
        ),
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
            evaluate_many(model, device, opponent, args.eval_seeds, args.eval_steps, args)
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1.0e-5)
    print(
        f"target_selection_mlp device={device} resumed={resumed} "
        f"iterations={args.iterations} rollout_steps={args.rollout_steps} "
            f"hidden_dim={args.hidden_dim} reward_mode={args.reward_mode} "
            f"terminal_win_bonus={args.terminal_win_bonus} "
            f"opponent_mix={args.opponent_mix} "
            f"selection_opponents={args.selection_opponents}",
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
            f"mean_shaping={rollout_metrics['mean_shaping_reward']:.3f} "
            f"mean_terminal_bonus={rollout_metrics['mean_terminal_bonus']:.4f} "
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
                    args,
                )
                eval_metrics[f"eval_{opponent}_mean_score_diff"] = opponent_metrics[
                    "mean_score_diff"
                ]
                eval_metrics[f"eval_{opponent}_min_score_diff"] = opponent_metrics[
                    "min_score_diff"
                ]
            if eval_metrics:
                selection_values = [
                    value
                    for key, value in eval_metrics.items()
                    if key.endswith("_mean_score_diff")
                    and key[len("eval_"):-len("_mean_score_diff")]
                    in set(args.selection_opponents)
                ]
                if not selection_values:
                    selection_values = [
                        value
                        for key, value in eval_metrics.items()
                        if key.endswith("_mean_score_diff")
                    ]
                eval_score = min(
                    selection_values
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
