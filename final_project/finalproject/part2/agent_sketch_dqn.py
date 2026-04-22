
from collections import deque
from types import SimpleNamespace

import numpy as np

from agents.agent_base import BaseAgent
from environments.collector.state import EnvState

# sep for dqn agent 
import torch 
from torch import nn

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
MAP_HEIGHT = 16
MAP_WIDTH = 16
MAX_DISTANCE = MAP_HEIGHT + MAP_WIDTH
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
    return int(rng.integers(4))

def valid_action_mask(tile_map, pos):
    mask = np.zeros(4, dtype=bool)
    for action, _ in valid_neighbors(tile_map, pos):
        mask[action] = True
    return mask


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


def shortest_item_distance(tile_map, start, item_positions):
    path = bfs_path_to_item(tile_map, start, item_positions)
    if not path:
        return None
    return len(path) - 1


def bfs_first_action_to_item(tile_map, start, item_positions):
    path = bfs_path_to_item(tile_map, start, item_positions)
    if len(path) < 2:
        return None

    next_pos = path[1]
    for action, candidate in valid_neighbors(tile_map, start):
        if candidate == next_pos:
            return action

    return None


def normalized_distance(distance):
    if distance is None:
        return 1.0
    return min(float(distance), float(MAX_DISTANCE)) / float(MAX_DISTANCE)


def action_planning_features(tile_map, self_pos, item_positions):
    valid_mask = valid_action_mask(tile_map, self_pos)
    current_distance = shortest_item_distance(tile_map, self_pos, item_positions)
    current_distance_value = MAX_DISTANCE if current_distance is None else current_distance

    next_distances = np.ones(ACTION_DIM, dtype=np.float32)
    distance_improvements = np.zeros(ACTION_DIM, dtype=np.float32)
    next_is_item = np.zeros(ACTION_DIM, dtype=np.float32)
    bfs_action_onehot = np.zeros(ACTION_DIM, dtype=np.float32)

    bfs_action = bfs_first_action_to_item(tile_map, self_pos, item_positions)
    if bfs_action is not None:
        bfs_action_onehot[bfs_action] = 1.0

    for action, next_pos in valid_neighbors(tile_map, self_pos):
        next_distance = shortest_item_distance(tile_map, next_pos, item_positions)
        next_distance_value = MAX_DISTANCE if next_distance is None else next_distance

        next_distances[action] = normalized_distance(next_distance)
        distance_improvements[action] = (
            float(current_distance_value - next_distance_value) / float(MAX_DISTANCE)
        )
        next_is_item[action] = float(tile_map[next_pos] == 2)

    return np.concatenate(
        [
            np.array([normalized_distance(current_distance)], dtype=np.float32),
            valid_mask.astype(np.float32),
            next_distances,
            distance_improvements,
            bfs_action_onehot,
            next_is_item,
        ]
    ).astype(np.float32)


def featurize_observation(observation):
    tile_map, self_pos, opponent_pos, item_positions = get_state_parts(observation)

    obstacle = (tile_map == 1).astype(np.float32)
    items = (tile_map == 2).astype(np.float32)

    self_layer = np.zeros_like(tile_map, dtype=np.float32)
    opp_layer = np.zeros_like(tile_map, dtype=np.float32)
    self_layer[self_pos] = 1.0
    opp_layer[opponent_pos] = 1.0
    team_points = np.asarray(observation["team_points"], dtype=np.float32)
    step_count = float(np.asarray(observation["steps"]).reshape(-1)[0])
    item_count = float(np.sum(items))

    scalar_features = np.array(
        [
            (team_points[0] - team_points[1]) / 100.0,
            team_points[0] / 100.0,
            team_points[1] / 100.0,
            step_count / 1000.0,
            item_count / 10.0,
        ],
        dtype=np.float32,
    )

    planning_features = action_planning_features(tile_map, self_pos, item_positions)

    features = np.concatenate([
        obstacle.reshape(-1),
        items.reshape(-1),
        self_layer.reshape(-1),
        opp_layer.reshape(-1),
        scalar_features,
        planning_features,
    ])

    mask = valid_action_mask(tile_map, self_pos)
    return features.astype(np.float32), mask

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, action_dim = 4):
        super().__init__()
        # def shared layers, want shared, value, advantage ...
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        # value net 
        self.value = nn.Linear(256, 1)
        # advantage net
        self.advantage = nn.Linear(256, action_dim)

    def forward(self, x):
        z = self.shared_layers(x)
        value = self.value(z)
        advantage = self.advantage(z)

        q = value + advantage - advantage.mean(dim=1, keepdim=True) # advantage.mean is baseline here 
        return q


FEATURE_DIM = MAP_HEIGHT * MAP_WIDTH * 4 + 5 + 21
    
class Agent(BaseAgent):
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.config = config

        self.device = torch.device("cpu")
        self.rng = np.random.default_rng(getattr(config, "seed", 0))

        self.q_network = DuelingDQN(
            input_dim=FEATURE_DIM,
            action_dim=ACTION_DIM,
        ).to(self.device)

    def load(self) -> None:
        # Later: load trained checkpoint here
        # checkpoint = torch.load(self.config.weights_path, map_location=self.device)
        # self.q_network.load_state_dict(checkpoint)
        self.q_network.eval()

    def act(self, observation: EnvState) -> int:
        features, valid_mask = featurize_observation(observation)

        x = torch.tensor(
            features,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(x).squeeze(0).cpu().numpy()

        q_values[~valid_mask] = -1e9

        return int(np.argmax(q_values))

