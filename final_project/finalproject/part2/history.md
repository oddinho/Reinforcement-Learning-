# Agent History

This file tracks attempted improvements to the Collector agent. Keep entries short and factual:
what changed, why it was tried, how it was evaluated, and what happened.

## Starting Point

### BFS Agent

- File: `part2/bfs_agent.py`
- Idea: Use breadth-first search to move along the shortest valid path to the nearest reachable item.
- Strength: Strong pathing because the map is fully observable and small (`16 x 16`).
- Weakness: Greedy target choice; does not reason deeply about opponent races, item clusters, or long-term positioning.
- Initial result:
  - vs random seed 0: `166 - 14`
  - vs baseline seed 0: `159 - 56`

### Simple Dueling DQN

- Files: `part2/agent_sketch_dqn.py`, `part2/train_agent.py`
- Input: flat map channels only.
  - obstacle layer
  - item layer
  - self-position layer
  - opponent-position layer
- Model: MLP dueling DQN.
- Training: replay buffer, epsilon-greedy actions, Double DQN target, Huber loss, gradient norm clipping.
- Issue: learned safe movement/loops more easily than reliable reward-seeking.

## Change Log

### 1. BFS Behavior Cloning

- Change: Added behavior cloning pretraining from BFS trajectories.
- Files: `train_agent.py`
- Motivation: Let the network imitate a strong planner before RL fine-tuning.
- Result:
  - Helped initialize behavior, but DQN fine-tuning could still drift away from BFS behavior.
- Notes:
  - Behavior cloning teaches action imitation, not direct value estimation.

### 2. Notebook-Friendly Training

- Change: `train_dqn()` can be called from notebook with config overrides.
- Added:
  - returned `history`
  - checkpoint helpers
  - checkpoint evaluation helpers
- Motivation: Easier plotting and experiment tracking.

### 3. Reward Scaling Attempt

- Change: Increased `reward_step_scale` from `1.0` to `5.0`.
- Motivation: Test whether stronger environment reward signal improves learning.
- Result:
  - Not clearly helpful.
  - Likely made movement too expensive because useful paths require many normal `-1` steps.
- Follow-up:
  - Reverted step scaling to `1.0`.

### 4. Distance-Based Reward Shaping

- Change: Added shortest-path distance shaping:
  - `reward += reward_distance_bonus * (prev_distance_to_item - next_distance_to_item)`
- Config after change:
  - `reward_step_scale = 1.0`
  - `reward_point_bonus = 20.0`
  - `reward_distance_bonus = 2.0`
- Motivation: Reward movement toward items before actual collection.
- Observed:
  - Training return started improving.
- Risk:
  - Shaped return may improve without improving final score.

### 5. DQN vs Baseline Replay Diagnostics

- Change: Generated latest DQN replays against baseline and random.
- Result example:
  - vs baseline seed 0: `10 - 18`
  - vs random seed 0: `2 - 6`
- Diagnosis:
  - No wall hits.
  - Still weak reward pursuit after early item collection.
  - Against random, policy showed mostly vertical movement.

### 6. BFS-Derived Planning Features

- Change: Expanded DQN input from `1024` to `1050` features.
- Added features:
  - scalar score/step/item-count features
  - current BFS distance to nearest item
  - valid action mask
  - BFS distance after each action
  - distance improvement after each action
  - BFS suggested action one-hot
  - whether next tile is an item
- Motivation: Let the model observe pathing information directly instead of trying to infer it from flat map channels.
- Consequence:
  - Old checkpoints with `FEATURE_DIM = 1024` no longer load into the new network.
  - Need to rerun BFS pretraining and DQN fine-tuning.

### 7. Latest DQN Replay Diagnostic

- Change: Recorded latest DQN checkpoint against pure BFS for visual inspection.
- Replay: `part2/replays/dqn_latest_vs_bfs_seed0.json`
- Checkpoint step: `30000`
- Latest checkpoint evaluation:
  - primary metric: `18.87`
  - baseline mean score diff: `+5.67`
  - random mean score diff: `+49.67`
- DQN vs BFS seed 0:
  - final score: `108 - 147`
  - score diff: `-39`
  - wall hits: `0`
  - item collection events: `108`
- Interpretation:
  - The learned DQN is now much stronger than earlier and can beat random/baseline on average.
  - Pure BFS still collects more efficiently in direct comparison.
  - Since there are no wall hits and many item events, remaining weakness is likely target choice/opponent race efficiency rather than basic movement validity.

### 8. Recent Position Features

- Change: Expanded DQN input from `1050` to `1054` features.
- Added features:
  - one feature per action indicating whether that action would move onto a recently visited tile
- Motivation:
  - Visualized DQN runs showed repeated up/down or left/right oscillation.
  - The previous flat observation had no memory, so the network could not directly tell whether a move was undoing a recent move.
- Implementation:
  - Training, behavior-cloning data collection, and checkpoint evaluation now carry a short recent-position window.
  - Default memory window: `8` previous positions.
- Consequence:
  - Old checkpoints with `FEATURE_DIM = 1050` no longer load into the new network.
  - Need to rerun BFS pretraining and DQN fine-tuning.
- Evaluation:
  - User reran training/evaluation and reported worse model quality than the previous `1050`-feature version.
- Decision:
  - Reverted from code.
  - Do not use recent-position memory as direct input for the next attempt.

### 9. Latest DQN Near-BFS Replay

- Change: Added a notebook-friendly replay helper:
  - `record_match(...)`
  - `record_checkpoint_match(...)`
- Motivation:
  - Visualize the latest trained DQN because it was close to pure BFS performance.
- Evaluation:
  - Checkpoint: `part2/checkpoints/dqn_latest.pt`
  - Checkpoint step: `30000`
  - Replay: `part2/replays/dqn_latest_vs_bfs_seed0.json`
  - vs BFS seed 0: `125 - 128`
  - score diff: `-3`
- Result:
  - Latest DQN is nearly matching BFS on this seed.
- Decision:
  - Visualize this replay before choosing the next algorithmic change.

### 10. Conservative BFS-Pretrained DQN Run

- Date: 2026-04-22
- Change:
  - Reran BFS behavior cloning followed by lower-learning-rate DQN fine-tuning.
- Motivation:
  - Preserve the useful BFS-pretrained policy while allowing DQN to improve from shaped reward.
  - Reduce policy drift by lowering RL learning rate and using lower epsilon.
- Pretraining config:
  - `num_steps = 25_000`
  - `epochs = 10`
  - `batch_size = 64`
  - `learning_rate = 1e-3`
  - `seed = 0`
  - `opponent = "random"`
- Pretraining result:
  - final BC epoch: `loss = 0.0064`, `accuracy = 0.997`
- DQN fine-tuning config:
  - `total_steps = 30_000`
  - `learning_starts = 2_000`
  - `batch_size = 64`
  - `learning_rate = 1e-4`
  - `epsilon_start = 0.15`
  - `epsilon_end = 0.02`
  - `epsilon_decay_steps = 15_000`
  - `target_update_interval = 2_000`
  - `eval_interval = 5_000`
  - `log_interval = 500`
  - `eval_seeds = (0, 1, 2)`
  - `opponent_mix = (("random", 1.0),)`
- Latest checkpoint:
  - checkpoint: `part2/checkpoints/dqn_latest.pt`
  - step: `30_000`
  - primary metric: `113.77`
  - baseline mean score diff: `+97.67`
  - random mean score diff: `+151.33`
- Best checkpoint:
  - checkpoint: `part2/checkpoints/dqn_best.pt`
  - step: `15_000`
  - primary metric: `121.13`
  - baseline mean score diff: `+109.33`
  - random mean score diff: `+148.67`
- BFS comparison:
  - latest checkpoint vs BFS seed 0: `125 - 128`
  - score diff: `-3`
  - replay: `part2/replays/dqn_latest_vs_bfs_seed0.json`
- Result:
  - Strongest DQN run so far against baseline/random.
  - Latest checkpoint nearly matches BFS on the inspected seed.
  - Best checkpoint by primary metric occurs earlier than final step, suggesting some mild late fine-tuning drift.
- Decision:
  - Keep this run as the current reference.
  - Next changes should be compared against both `dqn_best.pt` and `dqn_latest.pt`, not only final training return.

## Open Questions

- Does the network improve score after seeing BFS planning features?
- Should the learned model choose actions directly, or should it choose targets while BFS handles movement?
- Should replay be seeded with BFS transitions before DQN updates?
- Should a small behavior-cloning loss be kept during DQN fine-tuning?

## Experiment Template

# to self (ojay, from ojay) for next time. 
- Look at ways to value bigger clusters of items higher, as now the model might emulate bfs too much and therefore simply greedyly go towards nearest reward.
- add bfs and some version of previous dqn algo into rotation of opponents, will also have to tweak performance_metric to account for additional opponents and their strength?. 


### N. Short Name

- Date:
- Change:
- Motivation:
- Config:
- Evaluation:
  - vs random:
  - vs baseline:
  - vs BFS:
- Result:
- Decision:
