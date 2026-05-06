from collections import deque
from types import SimpleNamespace

import numpy as np

from agents.agent_base import BaseAgent
from environments.collector.state import EnvState


ACTION_DIM = 4
ACTIONS = np.array(
    [
        [-1, 0],
        [0, 1],
        [1, 0],
        [0, -1],
    ],
    dtype=np.int32,
)


DEFAULT_CONFIG = {
    "cluster_radius": 4,
    "cluster_weight": 0.75,
    "distance_weight": 1.0,
    "opponent_weight": 0.5,
    "route_weight": 10.0,
    "route_depth": 3,
    "route_decay": 0.7,
    "switch_margin": 1.5,
    "max_extra_distance": 3,
    "min_cluster_size": 2,
}


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


def route_value(target, items, item_distance_maps, depth, decay):
    current = target
    remaining = set(items)
    remaining.discard(target)
    value = 0.0

    for route_step in range(depth):
        distances = item_distance_maps[current]
        reachable = [
            (distances[other], other)
            for other in remaining
            if other in distances
        ]
        if not reachable:
            break

        leg_distance, next_item = min(reachable)
        value += (decay ** route_step) / float(1 + leg_distance)
        remaining.remove(next_item)
        current = next_item

    return value


def target_stats(tile_map, self_pos, opponent_pos, item_positions, config):
    own_distances, first_actions = bfs_distances_and_first_actions(tile_map, self_pos)
    opponent_distances, _ = bfs_distances_and_first_actions(tile_map, opponent_pos)
    items = [item for item in item_positions if item in own_distances]
    if not items:
        return []

    item_distance_maps = {
        item: bfs_distances_and_first_actions(tile_map, item)[0] for item in items
    }
    stats = []

    for item in items:
        own_distance = own_distances[item]
        opponent_distance = opponent_distances.get(item)
        race_margin = 0.0
        if opponent_distance is not None:
            race_margin = float(opponent_distance - own_distance)

        cluster_size = sum(
            1
            for other in items
            if item_distance_maps[item].get(other, 10_000) <= config["cluster_radius"]
        )
        local_route_value = route_value(
            target=item,
            items=items,
            item_distance_maps=item_distance_maps,
            depth=config["route_depth"],
            decay=config["route_decay"],
        )
        score = (
            config["cluster_weight"] * float(cluster_size)
            - config["distance_weight"] * float(own_distance)
            + config["opponent_weight"] * race_margin
            + config["route_weight"] * local_route_value
        )
        stats.append(
            {
                "target": item,
                "distance": int(own_distance),
                "first_action": first_actions[item],
                "cluster_size": int(cluster_size),
                "race_margin": float(race_margin),
                "route_value": float(local_route_value),
                "score": float(score),
            }
        )

    return stats


def choose_action(observation, config):
    tile_map, self_pos, opponent_pos, item_positions = get_state_parts(observation)
    stats = target_stats(tile_map, self_pos, opponent_pos, item_positions, config)
    if not stats:
        return None

    greedy = min(stats, key=lambda row: row["distance"])
    candidate = max(
        stats,
        key=lambda row: (
            row["score"],
            row["route_value"],
            row["cluster_size"],
            -row["distance"],
            row["race_margin"],
        ),
    )
    extra_distance = candidate["distance"] - greedy["distance"]
    score_edge = candidate["score"] - greedy["score"]

    should_switch = (
        candidate["target"] != greedy["target"]
        and candidate["cluster_size"] >= config["min_cluster_size"]
        and extra_distance <= config["max_extra_distance"]
        and score_edge >= config["switch_margin"]
        and candidate["first_action"] is not None
    )
    if should_switch:
        return int(candidate["first_action"])
    if greedy["first_action"] is not None:
        return int(greedy["first_action"])
    return None


class Agent(BaseAgent):
    """Deterministic BFS variant with gated route-value target switching.

    The agent normally follows nearest-item BFS. It switches to a nearby
    alternative target only when that target opens a better short route through
    follow-up items, matching the best `rollout_gated` autoresearch variant.
    """

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.config = dict(DEFAULT_CONFIG)
        for key in DEFAULT_CONFIG:
            if hasattr(config, key):
                self.config[key] = getattr(config, key)
        self.rng = np.random.default_rng(int(getattr(config, "seed", 0)))

    def load(self) -> None:
        pass

    def act(self, observation: EnvState) -> int:
        action = choose_action(observation, self.config)
        if action is not None:
            return int(action)

        tile_map, self_pos, _opponent_pos, _item_positions = get_state_parts(observation)
        return safe_random_action(tile_map, self_pos, self.rng)
