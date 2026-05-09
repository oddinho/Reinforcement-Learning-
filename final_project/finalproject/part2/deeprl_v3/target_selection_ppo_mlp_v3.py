"""Cluster-control target-selection policy used by PPO training and replay.

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
FEATURE_DIM = 31
GLOBAL_FEATURE_DIM = 19
ROUTE_DEPTH = 3
ROUTE_DECAY = 0.7

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
    global_features: np.ndarray
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


def get_score_context(observation):
    team_points = np.asarray(observation["team_points"], dtype=np.float32)
    steps = float(np.asarray(observation["steps"]))
    score_diff = float(team_points[0] - team_points[1])
    score_diff_norm = float(np.clip(score_diff / 50.0, -1.0, 1.0))
    behind = 1.0 if score_diff < 0.0 else 0.0
    steps_norm = float(np.clip(steps / 1000.0, 0.0, 1.0))
    steps_remaining_norm = 1.0 - steps_norm
    urgent_behind = behind * steps_norm
    return score_diff_norm, behind, steps_norm, steps_remaining_norm, urgent_behind


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


def nearby_cluster_items(item_distance_maps, target, items, radius):
    distances = item_distance_maps[target]
    return [item for item in items if distances.get(item, 10_000) <= radius]


def route_value(target, items, item_distance_maps, depth=ROUTE_DEPTH, decay=ROUTE_DECAY):
    """Cheap feature for whether a target opens a short follow-up route."""

    current = target
    remaining = set(items)
    remaining.discard(target)
    value = 0.0

    for route_step in range(depth):
        distances = item_distance_maps[current]
        reachable = [
            (distances[item], item)
            for item in remaining
            if item in distances
        ]
        if not reachable:
            break
        leg_distance, next_item = min(reachable)
        value += (decay ** route_step) / float(1 + leg_distance)
        remaining.remove(next_item)
        current = next_item

    return float(np.clip(value / 1.5, 0.0, 1.0))


def position_features(target, cluster_value):
    """Target-position hints that are cheap for a small MLP to consume."""

    y, x = target
    center_y = (MAP_HEIGHT - 1) / 2.0
    center_x = (MAP_WIDTH - 1) / 2.0
    max_center_distance = center_y + center_x
    center_distance = abs(float(y) - center_y) + abs(float(x) - center_x)
    center_score = 1.0 - center_distance / max_center_distance

    edge_distance = min(y, x, MAP_HEIGHT - 1 - y, MAP_WIDTH - 1 - x)
    max_edge_distance = max(1.0, float(min(MAP_HEIGHT, MAP_WIDTH) // 2))
    edge_score = 1.0 - min(float(edge_distance), max_edge_distance) / max_edge_distance

    center_cluster_value = center_score * cluster_value
    edge_cluster_value = edge_score * cluster_value
    return (
        float(np.clip(center_score, 0.0, 1.0)),
        float(np.clip(edge_score, 0.0, 1.0)),
        float(np.clip(center_cluster_value, 0.0, 1.0)),
        float(np.clip(edge_cluster_value, 0.0, 1.0)),
    )


def cluster_control_features(
    target,
    items,
    item_distance_maps,
    own_distances,
    opponent_distances,
    own_distance,
    opponent_distance,
    cluster_value,
    contested,
):
    """Estimate whether this target controls a local item region.

    This is deliberately cheap: use the BFS-radius-5 neighborhood around the
    target as the local cluster, then count how many items in that neighborhood
    our agent can reach no later than the opponent.
    """

    cluster_items = nearby_cluster_items(item_distance_maps, target, items, radius=5)
    if not cluster_items:
        return 0.0, 0.0, 0.0, 0.0

    controlled = 0
    margins = []
    own_sum = 0.0
    opponent_sum = 0.0
    for item in cluster_items:
        own_item_distance = own_distances.get(item, MAX_DISTANCE)
        opponent_item_distance = opponent_distances.get(item, MAX_DISTANCE)
        own_sum += float(own_item_distance)
        opponent_sum += float(opponent_item_distance)
        margin = float(opponent_item_distance - own_item_distance)
        margins.append(margin)
        if margin >= 0.0:
            controlled += 1

    cluster_size = max(1, len(cluster_items))
    cluster_control_value = min(float(controlled), 10.0) / 10.0
    cluster_race_margin = float(
        np.clip(np.mean(margins) / MAX_DISTANCE, -1.0, 1.0)
    )
    own_cluster_distance = own_sum / float(cluster_size)
    opponent_cluster_distance = opponent_sum / float(cluster_size)
    cluster_distance_margin = float(
        np.clip(
            (opponent_cluster_distance - own_cluster_distance) / MAX_DISTANCE,
            -1.0,
            1.0,
        )
    )

    opponent_distance_feature = (
        opponent_distance if opponent_distance is not None else MAX_DISTANCE
    )
    immediate_margin = float(opponent_distance_feature - own_distance)
    singleton_tunnel_flag = (
        1.0
        if contested >= 1.0
        and cluster_value < 0.25
        and abs(immediate_margin) <= 1.0
        else 0.0
    )
    cluster_swing_value = float(
        np.clip(
            cluster_control_value - 0.5 * singleton_tunnel_flag,
            -1.0,
            1.0,
        )
    )
    return (
        float(np.clip(cluster_control_value, 0.0, 1.0)),
        cluster_race_margin,
        singleton_tunnel_flag,
        cluster_swing_value,
    )


def build_global_features(
    self_pos,
    opponent_pos,
    items,
    own_target_distances,
    opponent_target_distances,
    race_margins,
    route_values,
    cluster3_values,
    cluster5_values,
    score_diff_norm,
    behind,
    steps_norm,
    steps_remaining_norm,
    urgent_behind,
):
    """State-level critic input.

    The actor still receives per-target rows. The critic gets these global
    scalars so its value estimate can distinguish easy/hard states without
    reconstructing that information from mean/max/min pooled target rows.
    """

    own_arr = np.asarray(own_target_distances, dtype=np.float32)
    opponent_arr = np.asarray(opponent_target_distances, dtype=np.float32)
    race_arr = np.asarray(race_margins, dtype=np.float32)
    route_arr = np.asarray(route_values, dtype=np.float32)
    cluster3_arr = np.asarray(cluster3_values, dtype=np.float32)
    cluster5_arr = np.asarray(cluster5_values, dtype=np.float32)
    contested = np.abs(race_arr) <= 2.0
    own_favored = race_arr >= 0.0
    return np.asarray(
        [
            (float(self_pos[0]) / (MAP_HEIGHT - 1)) * 2.0 - 1.0,
            (float(self_pos[1]) / (MAP_WIDTH - 1)) * 2.0 - 1.0,
            (float(opponent_pos[0]) / (MAP_HEIGHT - 1)) * 2.0 - 1.0,
            (float(opponent_pos[1]) / (MAP_WIDTH - 1)) * 2.0 - 1.0,
            min(len(items), 32) / 32.0,
            normalized_distance(float(np.min(own_arr))),
            normalized_distance(float(np.mean(own_arr))),
            normalized_distance(float(np.min(opponent_arr))),
            normalized_distance(float(np.mean(opponent_arr))),
            float(np.mean(contested.astype(np.float32))),
            float(np.mean(own_favored.astype(np.float32))),
            float(np.max(route_arr)),
            min(float(np.max(cluster3_arr)), 10.0) / 10.0,
            min(float(np.max(cluster5_arr)), 16.0) / 16.0,
            score_diff_norm,
            behind,
            steps_norm,
            steps_remaining_norm,
            urgent_behind,
        ],
        dtype=np.float32,
    )


def build_target_candidates(observation):
    tile_map, self_pos, opponent_pos, item_positions = get_state_parts(observation)
    (
        score_diff_norm,
        behind,
        steps_norm,
        steps_remaining_norm,
        urgent_behind,
    ) = get_score_context(observation)
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
    own_target_distances = []
    opponent_target_distances = []
    race_margins = []
    route_values = []
    cluster3_values = []
    cluster5_values = []
    for item in items:
        own_distance = own_distances[item]
        opponent_distance = opponent_distances.get(item)
        race_margin = 0.0
        if opponent_distance is not None:
            race_margin = float(opponent_distance - own_distance)

        cluster3 = count_cluster(item_distance_maps, item, items, radius=3)
        cluster5 = count_cluster(item_distance_maps, item, items, radius=5)
        local_route_value = route_value(item, items, item_distance_maps)
        cluster_value = 0.5 * (
            min(float(cluster3), 10.0) / 10.0
            + min(float(cluster5), 16.0) / 16.0
        )
        (
            center_score,
            edge_score,
            center_cluster_value,
            edge_cluster_value,
        ) = position_features(item, cluster_value)
        opponent_distance_feature = (
            opponent_distance if opponent_distance is not None else MAX_DISTANCE
        )
        contested = 1.0 if abs(race_margin) <= 2.0 else 0.0
        own_favored = 1.0 if race_margin >= 0.0 else 0.0
        lost_race = 1.0 if race_margin < -1.0 else 0.0
        (
            cluster_control_value,
            cluster_race_margin,
            singleton_tunnel_flag,
            cluster_swing_value,
        ) = cluster_control_features(
            target=item,
            items=items,
            item_distance_maps=item_distance_maps,
            own_distances=own_distances,
            opponent_distances=opponent_distances,
            own_distance=own_distance,
            opponent_distance=opponent_distance,
            cluster_value=cluster_value,
            contested=contested,
        )
        route_if_own_favored = local_route_value * own_favored
        route_if_lost = local_route_value * lost_race
        abandon_pressure = lost_race * (0.5 + 0.5 * behind)
        own_target_distances.append(float(own_distance))
        opponent_target_distances.append(float(opponent_distance_feature))
        race_margins.append(float(race_margin))
        route_values.append(float(local_route_value))
        cluster3_values.append(float(cluster3))
        cluster5_values.append(float(cluster5))
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
            local_route_value,
            item_count_norm,
            contested,
            own_favored,
            lost_race,
            route_if_own_favored,
            route_if_lost,
            score_diff_norm,
            behind,
            urgent_behind,
            abandon_pressure,
            center_score,
            edge_score,
            center_cluster_value,
            edge_cluster_value,
            cluster_control_value,
            cluster_race_margin,
            singleton_tunnel_flag,
            cluster_swing_value,
        ]
        features.append(row)
        targets.append(item)
        target_first_actions.append(first_actions[item])

    global_features = build_global_features(
        self_pos=self_pos,
        opponent_pos=opponent_pos,
        items=items,
        own_target_distances=own_target_distances,
        opponent_target_distances=opponent_target_distances,
        race_margins=race_margins,
        route_values=route_values,
        cluster3_values=cluster3_values,
        cluster5_values=cluster5_values,
        score_diff_norm=score_diff_norm,
        behind=behind,
        steps_norm=steps_norm,
        steps_remaining_norm=steps_remaining_norm,
        urgent_behind=urgent_behind,
    )

    return TargetCandidates(
        features=np.asarray(features, dtype=np.float32),
        global_features=global_features,
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
        critic_input_dim = self.feature_dim * 3 + GLOBAL_FEATURE_DIM + 1
        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features, valid_mask, global_features):
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
            [mean_features, max_features, min_features, global_features, count_feature],
            dim=1,
        )
        values = self.critic(summary).squeeze(-1)
        return logits, values


def make_single_batch(candidates, device):
    features = torch.from_numpy(candidates.features[None, :, :]).to(device)
    global_features = torch.from_numpy(candidates.global_features[None, :]).to(device)
    valid_mask = torch.ones(
        (1, candidates.features.shape[0]),
        dtype=torch.bool,
        device=device,
    )
    return features, valid_mask, global_features


@torch.no_grad()
def select_target_action(model, candidates, device, sample):
    model.eval()
    features, valid_mask, global_features = make_single_batch(candidates, device)
    logits, values = model(features, valid_mask, global_features)
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
