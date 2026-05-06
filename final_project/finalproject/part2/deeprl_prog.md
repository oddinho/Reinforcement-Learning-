# Deep RL Progress

## 2026-05-06 Reset

The previous target-selection PPO became too complex again: CNN board encoder,
behavior cloning, and many engineered route features made it harder to reason
about what PPO was actually learning.

The current direction scraps that implementation and resets to a simpler PPO
model inspired by the 2017 PPO paper:

```text
reachable target features -> Linear(64) -> Tanh -> Linear(64) -> Tanh -> logit
```

Training is now separate from model/inference code:

- `target_selection_ppo_mlp.py`: model, target features, BFS target-to-action.
- `train_target_selection_ppo.py`: PPO training and evaluation only.

The new default training opponent mix is:

```text
bfs:0.5,rollout_gated:0.5
```

The checkpoint name also changed to avoid confusing it with the older CNN/BC
checkpoint:

```text
checkpoints/target_selection_ppo_mlp_latest.pt
```

The older results below are kept as context, but they are no longer from the
active architecture.

Goal: train a neural target ranker that beats both nearest-item BFS and the
current deterministic `rollout_gated` submission agent on average, not just on
one lucky seed.

## Current Baselines

- `deterministic_agents/bfs_agent.py`: shortest path to the nearest item.
- `deterministic_agents/rollout_gated_agent.py`: BFS plus route-aware switching. This is the
  current strong hand-written baseline and the leaderboard candidate.

Any learned agent should be evaluated against both. Beating BFS alone is not
enough anymore.

## Current Learning Approach

The PPO problem is intentionally narrowed:

```text
neural network chooses target item -> BFS executes shortest path
```

This makes the learning problem target selection rather than movement. On a
known grid, BFS already solves movement optimally, so asking PPO to rediscover
pathfinding wastes samples.

## Active Experiment

Current run:

```text
rollout-gated behavior cloning -> PPO against bfs/rollout_gated mix
```

The behavior-cloning phase teaches the model the rollout-gated target choices.
PPO should then improve from that warm start.

Early signal from the active run:

- BC accuracy reached about `0.986`, so the model can imitate the teacher.
- PPO iteration 3 was slightly above BFS on the 5-seed eval.
- Policy entropy became very low, so the model may be too deterministic too
  early.

## Current Results

Active run command shape:

```text
pretrain_steps=10000, pretrain_epochs=5, iterations=20,
rollout_steps=3000, learning_rate=1e-4, entropy_coef=0.02,
opponent_mix=bfs:0.7,rollout_gated:0.3
```

Behavior cloning:

- collected `9982` target-choice examples
- final BC loss `0.0532`
- final BC accuracy `0.986`
- final BC entropy `0.060`

The training run finished at PPO iteration 20. Final 5-seed eval from the
training log:

| Opponent | Seeds | Mean score diff | Min | Max |
| --- | --- | ---: | ---: | ---: |
| BFS | `0..4` | `+6.60` | `-9` | `+39` |
| rollout_gated | `0..4` | `+3.40` | `-5` | `+10` |
| baseline | `0..4` | `+95.00` | `+77` | `+106` |

Follow-up eval-only run over 10 seeds:

| Opponent | Seeds | Mean score diff | Min | Max |
| --- | --- | ---: | ---: | ---: |
| BFS | `0..9` | `+6.10` | `-9` | `+39` |
| rollout_gated | `0..9` | `+2.70` | `-8` | `+11` |

Interpretation:

- The model now beats BFS on average over the 10-seed check.
- It also beats `rollout_gated` on average over the same seeds, which clears
  the first useful threshold.
- The margin is still small against `rollout_gated`, so this should be tested
  on more seeds before replacing the deterministic submission agent.
- The low entropy means the policy quickly becomes near-deterministic after
  BC, so future runs should either keep more exploration or use a weaker BC
  anchor rather than fully locking onto the teacher.

## Changes Added For Next Run

- Save a separate best checkpoint chosen by the weakest mean score difference
  across eval opponents.
- Add optional PPO-time BC anchoring with `--bc-anchor-coef`.

The best-checkpoint rule matters because baseline games are easy. The model
should be selected by performance against the hard opponents: BFS and
rollout-gated.

## Next Runs

Recommended command from `final_project/finalproject/part2`:

```powershell
python .\train_target_selection_ppo.py --pretrain-steps 10000 --pretrain-epochs 5 --iterations 30 --rollout-steps 3000 --batch-size 256 --ppo-epochs 4 --learning-rate 1e-4 --entropy-coef 0.03 --bc-anchor-coef 0.01 --opponent-mix bfs:0.6,rollout_gated:0.4 --eval-opponents bfs rollout_gated --eval-seeds 0 1 2 3 4 5 6 7 8 9 --eval-steps 1000 --checkpoint-path .\checkpoints\target_selection_ppo_bc_anchor_latest.pt
```

If entropy stays below about `0.05` and eval does not improve, use one of these
instead of increasing training time blindly:

- increase `--entropy-coef` to `0.05`
- reduce `--bc-anchor-coef` to `0.003`
- increase opponent pressure to `bfs:0.5,rollout_gated:0.5`

## Success Criterion

A candidate is worth packaging only if it has positive mean score difference
against both:

- BFS over at least 10 fixed seeds
- rollout-gated over at least 10 fixed seeds

Then record a replay for inspection and package:

```text
agent.py
config.yaml
weights/model.pth
```

Checkpoint eval command:

```powershell
python .\train_target_selection_ppo.py --eval-only --checkpoint-path .\checkpoints\target_selection_ppo_bc_latest.pt --eval-opponents bfs rollout_gated --eval-seeds 0 1 2 3 4 5 6 7 8 9 --eval-steps 1000
```
