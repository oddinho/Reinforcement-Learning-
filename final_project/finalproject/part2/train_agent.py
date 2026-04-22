
import json
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import random
import numpy as np
import torch
import torch.nn.functional as F

# keep training file sep from the agent file
PART2_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PART2_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agents.baseline.agent import Agent as BaselineAgent
from agents.random.agent import Agent as RandomAgent
from environments.collector.wrappers import CollectorGymEnv, RecordEpisode

import bfs_agent
from agent_sketch_dqn import (
    ACTION_DIM,
    FEATURE_DIM,
    DuelingDQN,
    featurize_observation,
)


DQN_CONFIG = {
    "seed": 0,
    "total_steps": 50000,
    "replay_capacity": 50_000,
    "learning_starts": 1000,
    "batch_size": 128,
    "gamma": 0.99,
    "learning_rate": 1e-3,
    "target_update_interval": 1_000,
    "eval_interval": 5000,
    "log_interval": 1000,
    "eval_seeds": (0, 1, 2),
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 15_000,
    "reward_point_bonus": 20.0,
    "reward_step_scale": 1.0,
    "reward_distance_bonus": 2.0,
    "opponent_mix": (
        ("random", 1.0),
    ),
}

CHECKPOINT_DIR = PART2_ROOT / "checkpoints"
REPLAY_DIR = PART2_ROOT / "replays"
LATEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "dqn_latest.pt"
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "dqn_best.pt"
BEST_METADATA_PATH = CHECKPOINT_DIR / "dqn_best.json"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_q_network(device=None):
    if device is None:
        device = get_device()
    return DuelingDQN(FEATURE_DIM, ACTION_DIM).to(device)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_valid_mask):
        self.buffer.append((
            state,
            action,
            reward,
            next_state,
            done,
            next_valid_mask,
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones, next_masks = zip(*batch)

        return (
            np.stack(states).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.array(dones, dtype=np.float32),
            np.stack(next_masks).astype(bool),
        )

    def __len__(self):
        return len(self.buffer)


def random_valid_action(valid_mask, rng, action_dim=4):
    valid_actions = np.flatnonzero(valid_mask)
    if len(valid_actions) > 0:
        return int(rng.choice(valid_actions))
    return int(rng.integers(action_dim))


def select_action(q_network, state, valid_mask, epsilon, rng, device):
    if rng.random() < epsilon:
        return random_valid_action(valid_mask, rng)

    state_t = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(device)

    q_network.eval()
    with torch.no_grad():
        q_values = q_network(state_t).squeeze(0).cpu().numpy()

    q_values[~valid_mask] = -1e9
    return int(np.argmax(q_values))

def dqn_update(q_network, target_network, optimizer, replay_buffer, batch_size, gamma, device):
    if len(replay_buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones, next_masks = replay_buffer.sample(batch_size)

    states_t = torch.from_numpy(states).to(device)
    actions_t = torch.from_numpy(actions).to(device)
    rewards_t = torch.from_numpy(rewards).to(device)
    next_states_t = torch.from_numpy(next_states).to(device)
    dones_t = torch.from_numpy(dones).to(device)
    next_masks_t = torch.from_numpy(next_masks).to(device)

    q_network.train()
    q_values = q_network(states_t)
    chosen_q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_online = q_network(next_states_t)
        next_q_online[~next_masks_t] = -1e9
        next_actions = next_q_online.argmax(dim=1)

        next_q_target = target_network(next_states_t)
        next_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q_values = rewards_t + gamma * next_values * (1.0 - dones_t)

    loss = F.smooth_l1_loss(chosen_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    # Clip gradients to hopefully stabilize training, think of it as: if len(vector(gradients)) > 1, then scale down to len ==1. 
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0) 
    optimizer.step()
    return float(loss.item())


def epsilon_by_step(step):
    progress = min(1.0, step / DQN_CONFIG["epsilon_decay_steps"])
    return DQN_CONFIG["epsilon_start"] + progress * (
        DQN_CONFIG["epsilon_end"] - DQN_CONFIG["epsilon_start"]
    )


def build_random_agent(seed):
    agent = RandomAgent(SimpleNamespace(seed=seed, action_space=ACTION_DIM))
    agent.load()
    return agent


def build_baseline_agent(seed):
    agent = BaselineAgent(SimpleNamespace(epsilon=0.3, seed=seed, action_space=ACTION_DIM))
    agent.load()
    return agent


def build_bfs_agent(seed):
    agent = bfs_agent.Agent(SimpleNamespace(seed=seed, action_space=ACTION_DIM))
    agent.load()
    return agent


def choose_training_opponent(rng, seed):
    names = [name for name, _weight in DQN_CONFIG["opponent_mix"]]
    weights = np.array([weight for _name, weight in DQN_CONFIG["opponent_mix"]], dtype=np.float64)
    weights = weights / weights.sum()
    name = names[int(rng.choice(len(names), p=weights))]
    if name == "baseline":
        return build_baseline_agent(seed), name
    return build_random_agent(seed), name


def nearest_item_distance(player_obs):
    tile_map, self_pos, _opponent_pos, item_positions = bfs_agent.get_state_parts(player_obs)
    path = bfs_agent.bfs_path_to_item(tile_map, self_pos, item_positions)
    if not path:
        return None
    return len(path) - 1


def shaped_reward(prev_player_obs, next_player_obs, reward):
    prev_points = np.asarray(prev_player_obs["team_points"], dtype=np.float32)
    next_points = np.asarray(next_player_obs["team_points"], dtype=np.float32)
    env_reward = np.asarray(reward, dtype=np.float32)

    own_point_delta = next_points[0] - prev_points[0]
    opponent_point_delta = next_points[1] - prev_points[1]
    point_delta = own_point_delta - opponent_point_delta
    step_reward = env_reward[0]
    prev_distance = nearest_item_distance(prev_player_obs)
    next_distance = nearest_item_distance(next_player_obs)

    distance_delta = 0.0
    if prev_distance is not None and next_distance is not None:
        distance_delta = float(prev_distance - next_distance)

    return float(
        DQN_CONFIG["reward_step_scale"] * step_reward
        + DQN_CONFIG["reward_point_bonus"] * point_delta
        + DQN_CONFIG["reward_distance_bonus"] * distance_delta
    )


class TrainedDQNAgent:
    def __init__(self, q_network, device):
        self.q_network = q_network
        self.device = device

    def reset(self):
        pass

    def load(self):
        pass

    def act(self, observation):
        state, valid_mask = featurize_observation(observation)
        return select_action(
            self.q_network,
            state,
            valid_mask,
            epsilon=0.0,
            rng=np.random.default_rng(0),
            device=self.device,
        )


def run_match(q_network, opponent_factory, seed, device, max_steps=1000):
    env = CollectorGymEnv(numpy_output=True)
    agent = TrainedDQNAgent(q_network, device)
    agent.reset()
    opponent = opponent_factory(seed)
    if hasattr(opponent, "reset"):
        opponent.reset()

    obs, _info = env.reset(seed=seed)
    total_reward = np.zeros(2, dtype=np.float32)
    steps = 0

    q_network.eval()
    for _ in range(max_steps):
        action = agent.act(obs["player_0"])
        opponent_action = opponent.act(obs["player_1"])
        obs, reward, terminated, truncated, info = env.step(
            {"player_0": action, "player_1": opponent_action}
        )
        total_reward += np.asarray(reward, dtype=np.float32)
        steps += 1
        done = bool(np.asarray(terminated).any() or np.asarray(truncated).any())
        if done:
            break

    points = np.asarray(info["state"].team_points, dtype=int)
    env.close()
    return {
        "seed": int(seed),
        "steps": int(steps),
        "team_points": points.tolist(),
        "score_diff": int(points[0] - points[1]),
        "reward_diff": float(total_reward[0] - total_reward[1]),
    }


def opponent_factory_by_name(name):
    if name == "baseline":
        return build_baseline_agent
    if name == "random":
        return build_random_agent
    if name == "bfs":
        return build_bfs_agent
    raise ValueError(f"Unknown opponent: {name}")


def record_match(q_network, opponent="bfs", seed=0, device=None, output_path=None, max_steps=1000):
    if device is None:
        device = get_device()
    if output_path is None:
        REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        output_path = REPLAY_DIR / f"dqn_latest_vs_{opponent}_seed{seed}.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = RecordEpisode(
        CollectorGymEnv(numpy_output=True),
        save_on_reset=False,
        save_on_close=False,
    )
    agent = TrainedDQNAgent(q_network, device)
    opponent_agent = opponent_factory_by_name(opponent)(seed)

    if hasattr(agent, "reset"):
        agent.reset()
    if hasattr(opponent_agent, "reset"):
        opponent_agent.reset()

    obs, _info = env.reset(seed=seed)
    total_reward = np.zeros(2, dtype=np.float32)
    steps = 0
    info = None

    q_network.eval()
    for _ in range(max_steps):
        action = agent.act(obs["player_0"])
        opponent_action = opponent_agent.act(obs["player_1"])
        obs, reward, terminated, truncated, info = env.step(
            {"player_0": action, "player_1": opponent_action}
        )
        total_reward += np.asarray(reward, dtype=np.float32)
        steps += 1
        done = bool(np.asarray(terminated).any() or np.asarray(truncated).any())
        if done:
            break

    env.save_episode(str(output_path))
    env.close()

    points = np.asarray(info["state"].team_points, dtype=int)
    return {
        "seed": int(seed),
        "opponent": opponent,
        "steps": int(steps),
        "team_points": points.tolist(),
        "score_diff": int(points[0] - points[1]),
        "reward_diff": float(total_reward[0] - total_reward[1]),
        "replay_path": str(output_path),
    }


def evaluate(q_network, device):
    q_network.eval()
    baseline_matches = [
        run_match(q_network, build_baseline_agent, seed, device)
        for seed in DQN_CONFIG["eval_seeds"]
    ]
    random_matches = [
        run_match(q_network, build_random_agent, seed, device)
        for seed in DQN_CONFIG["eval_seeds"]
    ]

    baseline_mean = float(np.mean([m["score_diff"] for m in baseline_matches]))
    random_mean = float(np.mean([m["score_diff"] for m in random_matches]))
    primary_metric = 0.7 * baseline_mean + 0.3 * random_mean

    return {
        "primary_metric": float(primary_metric),
        "baseline_mean_score_diff": baseline_mean,
        "random_mean_score_diff": random_mean,
        "baseline_matches": baseline_matches,
        "random_matches": random_matches,
    }


def evaluate_vs_bfs(q_network, device=None, seeds=None):
    if device is None:
        device = get_device()
    if seeds is None:
        seeds = DQN_CONFIG["eval_seeds"]

    matches = [
        run_match(q_network, build_bfs_agent, seed, device)
        for seed in seeds
    ]
    return {
        "mean_score_diff": float(np.mean([m["score_diff"] for m in matches])),
        "mean_reward_diff": float(np.mean([m["reward_diff"] for m in matches])),
        "matches": matches,
    }


def evaluate_checkpoint_vs_bfs(path=LATEST_CHECKPOINT_PATH, seeds=None):
    device = get_device()
    q_network, checkpoint = load_checkpoint(path, device)
    return evaluate_vs_bfs(q_network, device, seeds), checkpoint


def record_checkpoint_match(
    path=LATEST_CHECKPOINT_PATH,
    opponent="bfs",
    seed=0,
    output_path=None,
    max_steps=1000,
):
    device = get_device()
    q_network, checkpoint = load_checkpoint(path, device)
    result = record_match(
        q_network,
        opponent=opponent,
        seed=seed,
        device=device,
        output_path=output_path,
        max_steps=max_steps,
    )
    return result, checkpoint


def save_checkpoint(q_network, path, step, evaluation=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": q_network.state_dict(),
            "feature_dim": FEATURE_DIM,
            "action_dim": ACTION_DIM,
            "step": int(step),
            "evaluation": evaluation,
        },
        path,
    )


def load_checkpoint(path=LATEST_CHECKPOINT_PATH, device=None):
    if device is None:
        device = get_device()
    path = Path(path)
    checkpoint = torch.load(path, map_location=device)
    q_network = make_q_network(device)
    q_network.load_state_dict(checkpoint["model_state_dict"])
    q_network.eval()
    return q_network, checkpoint


def evaluate_checkpoint(path=LATEST_CHECKPOINT_PATH, eval_seeds=None):
    device = get_device()
    if eval_seeds is None:
        q_network, checkpoint = load_checkpoint(path, device)
        return evaluate(q_network, device), checkpoint

    old_eval_seeds = DQN_CONFIG["eval_seeds"]
    DQN_CONFIG["eval_seeds"] = tuple(eval_seeds)
    try:
        q_network, checkpoint = load_checkpoint(path, device)
        return evaluate(q_network, device), checkpoint
    finally:
        DQN_CONFIG["eval_seeds"] = old_eval_seeds


def collect_bfs_dataset(num_steps, seed=0, opponent="random"):
    rng = np.random.default_rng(seed)
    env = CollectorGymEnv(numpy_output=True)
    teacher = bfs_agent.Agent(SimpleNamespace(seed=seed, action_space=ACTION_DIM))
    teacher.load()

    states = []
    actions = []
    masks = []
    obs, _info = env.reset(seed=seed)
    opponent_agent = build_baseline_agent(seed) if opponent == "baseline" else None

    for step in range(num_steps):
        player_obs = obs["player_0"]
        state, valid_mask = featurize_observation(player_obs)
        action = teacher.act(player_obs)

        states.append(state)
        actions.append(action)
        masks.append(valid_mask)

        if opponent_agent is not None:
            opponent_action = opponent_agent.act(obs["player_1"])
        else:
            opponent_action = int(rng.integers(ACTION_DIM))

        obs, _reward, terminated, truncated, _info = env.step(
            {"player_0": action, "player_1": opponent_action}
        )
        done = bool(np.asarray(terminated).any() or np.asarray(truncated).any())
        if done:
            obs, _info = env.reset(seed=seed + step + 1)
            if opponent_agent is not None:
                opponent_agent = build_baseline_agent(seed + step + 1)

    env.close()
    return (
        np.stack(states).astype(np.float32),
        np.asarray(actions, dtype=np.int64),
        np.stack(masks).astype(bool),
    )


def pretrain_from_bfs_dataset(
    q_network,
    states,
    actions,
    masks,
    epochs=5,
    batch_size=128,
    learning_rate=1e-3,
    device=None,
    progress=True,
):
    if device is None:
        device = get_device()

    q_network.to(device)
    optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
    states_t = torch.from_numpy(states.astype(np.float32)).to(device)
    actions_t = torch.from_numpy(actions.astype(np.int64)).to(device)
    masks_t = torch.from_numpy(masks.astype(bool)).to(device)
    sample_count = len(actions)
    history = []

    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(sample_count, device=device)
        losses = []
        accuracies = []

        q_network.train()
        for start in range(0, sample_count, batch_size):
            idx = permutation[start:start + batch_size]
            logits = q_network(states_t[idx])
            logits = logits.masked_fill(~masks_t[idx], -1e9)
            loss = F.cross_entropy(logits, actions_t[idx])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == actions_t[idx]).float().mean()

            losses.append(float(loss.item()))
            accuracies.append(float(acc.item()))

        row = {
            "epoch": int(epoch),
            "loss": float(np.mean(losses)),
            "accuracy": float(np.mean(accuracies)),
        }
        history.append(row)
        if progress:
            print(
                f"bc epoch={epoch} "
                f"loss={row['loss']:.4f} "
                f"accuracy={row['accuracy']:.3f}"
            )

    q_network.eval()
    return history


def pretrain_from_bfs(
    num_steps=10_000,
    epochs=5,
    batch_size=128,
    learning_rate=1e-3,
    seed=0,
    opponent="random",
    progress=True,
):
    device = get_device()
    q_network = make_q_network(device)
    states, actions, masks = collect_bfs_dataset(
        num_steps=num_steps,
        seed=seed,
        opponent=opponent,
    )
    history = pretrain_from_bfs_dataset(
        q_network,
        states,
        actions,
        masks,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
        progress=progress,
    )
    evaluation = evaluate(q_network, device)
    save_checkpoint(q_network, LATEST_CHECKPOINT_PATH, step=0, evaluation=evaluation)
    return {
        "q_network": q_network,
        "state_dict": q_network.state_dict(),
        "history": history,
        "evaluation": evaluation,
        "dataset_size": int(len(actions)),
        "latest_checkpoint_path": str(LATEST_CHECKPOINT_PATH),
    }


def save_best_metadata(step, evaluation):
    BEST_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BEST_METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump({"step": int(step), **evaluation}, f, indent=2)


def train_dqn(config_updates=None, progress=True, initial_state_dict=None):
    old_config = DQN_CONFIG.copy()
    if config_updates:
        DQN_CONFIG.update(config_updates)

    rng = np.random.default_rng(DQN_CONFIG["seed"])
    random.seed(DQN_CONFIG["seed"])
    torch.manual_seed(DQN_CONFIG["seed"])
    device = get_device()

    q_network = make_q_network(device)
    if initial_state_dict is not None:
        q_network.load_state_dict(initial_state_dict)
    target_network = make_q_network(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = torch.optim.Adam(q_network.parameters(), lr=DQN_CONFIG["learning_rate"])
    replay_buffer = ReplayBuffer(DQN_CONFIG["replay_capacity"])
    env = CollectorGymEnv(numpy_output=True)

    obs = None
    opponent = None
    episode_seed = DQN_CONFIG["seed"]
    episode_return = 0.0
    episode_count = 0
    recent_losses = deque(maxlen=200)
    recent_episode_returns = deque(maxlen=50)
    best_metric = float("-inf")
    history = []

    try:
        for step in range(1, DQN_CONFIG["total_steps"] + 1):
            if obs is None:
                opponent, _opponent_name = choose_training_opponent(rng, episode_seed)
                if hasattr(opponent, "reset"):
                    opponent.reset()
                obs, _info = env.reset(seed=episode_seed)
                episode_seed += 1
                episode_return = 0.0
                episode_count += 1

            state, valid_mask = featurize_observation(obs["player_0"])
            epsilon = epsilon_by_step(step)
            action = select_action(q_network, state, valid_mask, epsilon, rng, device)
            opponent_action = opponent.act(obs["player_1"])
            prev_player_obs = obs["player_0"]

            next_obs, reward, terminated, truncated, _info = env.step(
                {"player_0": action, "player_1": opponent_action}
            )
            done = bool(np.asarray(terminated).any() or np.asarray(truncated).any())
            next_state, next_valid_mask = featurize_observation(next_obs["player_0"])
            train_reward = shaped_reward(prev_player_obs, next_obs["player_0"], reward)
            episode_return += train_reward

            replay_buffer.push(
                state,
                action,
                train_reward,
                next_state,
                done,
                next_valid_mask,
            )

            loss = None
            if len(replay_buffer) >= DQN_CONFIG["learning_starts"]:
                loss = dqn_update(
                    q_network,
                    target_network,
                    optimizer,
                    replay_buffer,
                    DQN_CONFIG["batch_size"],
                    DQN_CONFIG["gamma"],
                    device,
                )
                if loss is not None:
                    recent_losses.append(loss)

            if step % DQN_CONFIG["target_update_interval"] == 0:
                target_network.load_state_dict(q_network.state_dict())

            if done:
                recent_episode_returns.append(episode_return)
                obs = None
            else:
                obs = next_obs

            if step % DQN_CONFIG["log_interval"] == 0:
                mean_loss = float(np.mean(recent_losses)) if recent_losses else float("nan")
                mean_return = (
                    float(np.mean(recent_episode_returns))
                    if recent_episode_returns
                    else float("nan")
                )
                row = {
                    "type": "train",
                    "step": int(step),
                    "epsilon": float(epsilon),
                    "buffer_size": int(len(replay_buffer)),
                    "episodes": int(episode_count),
                    "mean_loss": mean_loss,
                    "mean_episode_return": mean_return,
                }
                history.append(row)
                if progress:
                    print(
                        f"step={step} "
                        f"epsilon={epsilon:.3f} "
                        f"buffer={len(replay_buffer)} "
                        f"episodes={episode_count} "
                        f"loss={mean_loss:.4f} "
                        f"return={mean_return:.2f}"
                    )

            if step % DQN_CONFIG["eval_interval"] == 0:
                evaluation = evaluate(q_network, device)
                save_checkpoint(q_network, LATEST_CHECKPOINT_PATH, step, evaluation)
                saved_best = False
                if evaluation["primary_metric"] > best_metric:
                    best_metric = evaluation["primary_metric"]
                    saved_best = True
                    save_checkpoint(q_network, BEST_CHECKPOINT_PATH, step, evaluation)
                    save_best_metadata(step, evaluation)
                history.append(
                    {
                        "type": "eval",
                        "step": int(step),
                        "saved_best": bool(saved_best),
                        **evaluation,
                    }
                )
                if progress:
                    print(
                        f"eval step={step} "
                        f"metric={evaluation['primary_metric']:.2f} "
                        f"baseline={evaluation['baseline_mean_score_diff']:.2f} "
                        f"random={evaluation['random_mean_score_diff']:.2f} "
                        f"best={best_metric:.2f}"
                    )

        final_eval = evaluate(q_network, device)
        save_checkpoint(q_network, LATEST_CHECKPOINT_PATH, DQN_CONFIG["total_steps"], final_eval)
        return {
            "final_evaluation": final_eval,
            "history": history,
            "device": str(device),
            "latest_checkpoint_path": str(LATEST_CHECKPOINT_PATH),
            "best_checkpoint_path": str(BEST_CHECKPOINT_PATH),
            "best_metric": None if best_metric == float("-inf") else float(best_metric),
        }
    finally:
        env.close()
        if config_updates:
            DQN_CONFIG.clear()
            DQN_CONFIG.update(old_config)


if __name__ == "__main__":
    result = train_dqn()
    print("Final evaluation:", result["final_evaluation"])
