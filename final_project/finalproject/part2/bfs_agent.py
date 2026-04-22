from collections import deque
from types import SimpleNamespace

import numpy as np

from agents.agent_base import BaseAgent
from environments.collector.state import EnvState


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

ACTIONS = np.array(
    [
        [-1, 0],
        [0, 1],
        [1, 0],
        [0, -1],
    ],
    dtype=np.int32,
)


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


def bfs_path_to_item(tile_map, start, item_positions):
    if not item_positions:
        return []

    targets = set(item_positions)
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        pos, path = queue.popleft()
        if pos in targets:
            return path

        for _action, next_pos in valid_neighbors(tile_map, pos):
            if next_pos in visited:
                continue
            visited.add(next_pos)
            queue.append((next_pos, path + [next_pos]))

    return []


def bfs_first_action_to_item(tile_map, start, item_positions):
    path = bfs_path_to_item(tile_map, start, item_positions)
    if len(path) < 2:
        return None

    next_pos = path[1]
    for action, candidate in valid_neighbors(tile_map, start):
        if candidate == next_pos:
            return action

    return None


def safe_random_action(tile_map, pos, rng):
    actions = [action for action, _ in valid_neighbors(tile_map, pos)]
    if actions:
        return int(rng.choice(actions))
    return int(rng.integers(4))


class Agent(BaseAgent):
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.config = config
        self.rng = np.random.default_rng(int(getattr(config, "seed", 0)))

    def load(self) -> None:
        pass

    def act(self, observation: EnvState) -> int:
        tile_map, self_pos, _opponent_pos, item_positions = get_state_parts(observation)
        action = bfs_first_action_to_item(tile_map, self_pos, item_positions)
        if action is not None:
            return int(action)
        return safe_random_action(tile_map, self_pos, self.rng)
