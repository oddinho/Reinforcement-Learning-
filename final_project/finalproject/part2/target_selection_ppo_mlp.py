"""Small target-selection policy used by PPO training and replay.

The policy does not learn movement. It scores reachable item targets, and BFS
turns the selected target into the first shortest-path move.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


ACTION_DIM = 4
MAP_HEIGHT = 16
MAP_WIDTH = 16
MAX_DISTANCE = MAP_HEIGHT + MAP_WIDTH
FEATURE_DIM = 13

ACTIONS = np.array(
    [
        [-1, 0],
        [0, 1],
        [1, 0],
        [0, -1],
    ],
    dtype=np.int32,
)


@dataclass
class TargetCandidates:
    features: np.ndarray
    targets: list[tuple[int, int]]
    first_actions: list[int | None]


def get_state_parts(observation):
    tile_map = np.asarray(observation["map_features"]["tile_type"])
    positions = np.asarray(observation["units"]["position"])
    self_pos = tuple(int(v) for v in positions[0])
    opponent_pos = tuple(int(v) for v in positions[1])
    item_positions = [
        tuple(int(v) for v in pos) for pos in np.argwhere(tile_map == 2)
    ]
    return tile_map, self_pos, opponent_pos, item_positions


def valid_neighbors(tile_map, pos):
    height, width = tile_map.shape
    y, x = pos
    for action, (dy, dx) in enumerate(ACTIONS):
        ny, nx = y + int(dy), x + int(dx)
        if 0 <= ny < height and 0 <= nx < width and tile_map[ny, nx] != 1:
            yield action, (ny, nx)


def safe_random_action(tile_map, pos, rng):
    actions = [action for action, _ in valid_neighbors(tile_map, pos)]
    if actions:
        return int(rng.choice(actions))
    return int(rng.integers(ACTION_DIM))


def bfs_distances_and_first_actions(tile_map, start):
    queue = deque([start])
    distances = {start: 0}
    first_actions = {start: None}

    while queue:
        pos = queue.popleft()
        for action, next_pos in valid_neighbors(tile_map, pos):
            if next_pos in distances:
                continue
            distances[next_pos] = distances[pos] + 1
            first_actions[next_pos] = (
                action if first_actions[pos] is None else first_actions[pos]
            )
            queue.append(next_pos)

    return distances, first_actions


def normalized_distance(distance):
    if distance is None:
        return 1.0
    return min(float(distance), float(MAX_DISTANCE)) / float(MAX_DISTANCE)


def count_cluster(item_distance_maps, target, items, radius):
    distances = item_distance_maps[target]
    return sum(1 for item in items if distances.get(item, 10_000) <= radius)


def build_target_candidates(observation):
    tile_map, self_pos, opponent_pos, item_positions = get_state_parts(observation)
    own_distances, first_actions = bfs_distances_and_first_actions(tile_map, self_pos)
    opponent_distances, _ = bfs_distances_and_first_actions(tile_map, opponent_pos)
    items = [item for item in item_positions if item in own_distances]
    if not items:
        return None

    item_distance_maps = {
        item: bfs_distances_and_first_actions(tile_map, item)[0] for item in items
    }
    nearest_distance = min(own_distances[item] for item in items)
    item_count_norm = min(len(items), 32) / 32.0

    features = []
    targets = []
    target_first_actions = []
    for item in items:
        own_distance = own_distances[item]
        opponent_distance = opponent_distances.get(item)
        race_margin = 0.0
        if opponent_distance is not None:
            race_margin = float(opponent_distance - own_distance)

        cluster3 = count_cluster(item_distance_maps, item, items, radius=3)
        cluster5 = count_cluster(item_distance_maps, item, items, radius=5)
        y, x = item
        row = [
            (float(y) / (MAP_HEIGHT - 1)) * 2.0 - 1.0,
            (float(x) / (MAP_WIDTH - 1)) * 2.0 - 1.0,
            float(y - self_pos[0]) / (MAP_HEIGHT - 1),
            float(x - self_pos[1]) / (MAP_WIDTH - 1),
            float(y - opponent_pos[0]) / (MAP_HEIGHT - 1),
            float(x - opponent_pos[1]) / (MAP_WIDTH - 1),
            normalized_distance(own_distance),
            normalized_distance(opponent_distance),
            float(np.clip(race_margin / MAX_DISTANCE, -1.0, 1.0)),
            1.0 if own_distance == nearest_distance else 0.0,
            min(float(cluster3), 10.0) / 10.0,
            min(float(cluster5), 16.0) / 16.0,
            item_count_norm,
        ]
        features.append(row)
        targets.append(item)
        target_first_actions.append(first_actions[item])

    return TargetCandidates(
        features=np.asarray(features, dtype=np.float32),
        targets=targets,
        first_actions=target_first_actions,
    )


class TargetSelectionMLP(nn.Module):
    """Two-hidden-layer tanh MLP target scorer.

    The actor is shared across targets: every candidate item is represented by
    the same small feature vector and receives one logit. The critic receives a
    pooled summary of the current candidate set.
    """

    def __init__(self, feature_dim=FEATURE_DIM, hidden_dim=64):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.actor = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        critic_input_dim = self.feature_dim * 3 + 1
        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features, valid_mask):
        logits = self.actor(features).squeeze(-1)
        logits = logits.masked_fill(~valid_mask, -1.0e9)

        mask = valid_mask.unsqueeze(-1)
        masked = features.masked_fill(~mask, 0.0)
        counts = mask.sum(dim=1).clamp(min=1).float()
        mean_features = masked.sum(dim=1) / counts

        max_features = features.masked_fill(~mask, -1.0e9).max(dim=1).values
        min_features = features.masked_fill(~mask, 1.0e9).min(dim=1).values
        count_feature = (counts / 32.0).clamp(max=1.0)
        summary = torch.cat(
            [mean_features, max_features, min_features, count_feature],
            dim=1,
        )
        values = self.critic(summary).squeeze(-1)
        return logits, values


def make_single_batch(candidates, device):
    features = torch.from_numpy(candidates.features[None, :, :]).to(device)
    valid_mask = torch.ones(
        (1, candidates.features.shape[0]),
        dtype=torch.bool,
        device=device,
    )
    return features, valid_mask


@torch.no_grad()
def select_target_action(model, candidates, device, sample):
    model.eval()
    features, valid_mask = make_single_batch(candidates, device)
    logits, values = model(features, valid_mask)
    dist = Categorical(logits=logits)
    if sample:
        action_index = dist.sample()
    else:
        action_index = torch.argmax(logits, dim=1)

    idx = int(action_index.item())
    action = candidates.first_actions[idx]
    if action is None:
        action = 0
    return (
        idx,
        int(action),
        float(dist.log_prob(action_index).item()),
        float(values.item()),
    )


class DeterministicTargetSelectionAgent:
    """Evaluation wrapper: choose the highest-scoring target, then BFS move."""

    def __init__(self, model, device, seed=0):
        self.model = model
        self.device = device
        self.rng = np.random.default_rng(seed)

    def act(self, observation):
        candidates = build_target_candidates(observation)
        if candidates is None:
            tile_map, self_pos, _opponent_pos, _items = get_state_parts(observation)
            return safe_random_action(tile_map, self_pos, self.rng)
        _idx, action, _log_prob, _value = select_target_action(
            self.model,
            candidates,
            self.device,
            sample=False,
        )
        return int(action)
