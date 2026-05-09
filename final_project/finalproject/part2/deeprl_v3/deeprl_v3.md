# Deep RL v3

This folder is an isolated v3 experiment. It does not modify the earlier PPO
or `deeprl_v2` files.

## Motivation

Replay inspection showed a specific failure mode:

```text
the agent keeps contesting tied/single targets instead of abandoning them for
a nearby cluster it can control before the opponent
```

The v2 model saw local cluster counts and route value, but it did not directly
represent cluster ownership. v3 keeps the simpler 64-hidden architecture and
adds features/reward terms that express cluster control.

## Architecture

The policy is still target selection plus BFS movement:

```text
observation -> reachable item targets -> MLP chooses target -> BFS moves
```

Actor:

```text
target_features -> Linear(64) -> Tanh -> Linear(64) -> Tanh -> Linear(1)
```

Critic:

```text
mean(target_features),
max(target_features),
min(target_features),
global_state_features,
candidate_count
```

Defaults return to the faster setup:

```text
hidden_dim=64
rollout_steps=3000
batch_size=256
league_iterations=20
```

## Cluster-Control Features

v3 expands the target feature vector from 27 to 31 dimensions. The original 27
features are kept in the same order. The new appended features are:

| Feature | Meaning |
| --- | --- |
| `cluster_control_value` | normalized count of nearby radius-5 cluster items our agent reaches no later than the opponent |
| `cluster_race_margin` | mean BFS race margin over the target's radius-5 cluster |
| `singleton_tunnel_flag` | selected target looks like a contested low-cluster singleton |
| `cluster_swing_value` | cluster-control value minus singleton tunnel pressure |

The point is to tell the actor:

```text
this target is a gateway into a cluster we can control
```

rather than only:

```text
this target has nearby items
```

## Reward

The default reward mode is:

```text
score_delta_cluster
```

Base reward remains:

```text
new_score_diff - previous_score_diff
```

The terminal win bonus remains:

```text
if final_score_diff > 0:
    reward += 5.0
```

The cluster signal is now:

```text
+ center-cluster bonus
+ route bonus
+ cluster-control bonus
+ positive cluster-swing bonus
- singleton-tunnel penalty
- lost-cluster penalty
```

Default coefficients:

| Term | Default |
| --- | ---: |
| `--cluster-signal-center-bonus` | `0.08` |
| `--cluster-signal-route-bonus` | `0.06` |
| `--cluster-signal-control-bonus` | `0.12` |
| `--cluster-signal-swing-bonus` | `0.08` |
| `--singleton-tunnel-penalty` | `0.08` |
| `--cluster-signal-lost-penalty` | `0.08` |
| `--cluster-signal-max-abs` | `0.20` |

The shaping is clipped so the score objective still dominates.

## PPO Opponent Pool

Default training opponent mix:

```text
bfs:0.15
rollout_gated:0.25
ppo:0.25
position_aware_ppo:0.35
```

`ppo` and `position_aware_ppo` are frozen checkpoints from the earlier work.

## League Training

`league_train_v3.py` runs repeated generations:

```text
train candidate
10-seed gate eval
accept if it improves enough
append accepted checkpoint as frozen league opponent
start next generation
```

Important v3 fix:

```text
seed = base_seed + generation * seed_stride
```

Rejected generations no longer repeat the exact same training run.

Generation 1 resumes from:

```text
../checkpoints/target_selection_ppo_mlp_position_selfplay_latest_best.pt
```

If a v3 candidate is accepted, later generations resume from the latest
accepted v3 checkpoint.

Accepted v3 snapshots are kept as frozen league opponents. The fixed baselines
still get most of the probability mass, while the accepted league bucket gets
`0.35` by default. Inside that league bucket, newer snapshots are weighted
slightly higher with recency decay `0.75`; older accepted snapshots stay in the
pool to reduce overfitting to only the newest opponent.

Default gate:

```text
win at least 7/10 seeds against rollout_gated
win at least 7/10 seeds against position_aware_ppo
mean_score_diff >= +5.0 against BFS
mean_score_diff >= +5.0 against rollout_gated
mean_score_diff >= +5.0 against old PPO
```

Training-time checkpoint selection still uses five eval seeds by default. The
accept/reject league gate uses ten seeds because the win criterion is 7/10.

Training-time best-checkpoint selection for the manual league runs uses the
latest PPO references for wins, and all evaluated opponents for mean score:

```text
win opponents: position_aware_ppo and league
win at least 3/5 seeds against both
mean opponents: BFS, rollout_gated, old PPO, position_aware_ppo, league
mean_score_diff > 0 against all mean opponents
```

With the current manual pool, `league` means `ppo_league_gen1.pt`.

## Commands

Single v3 training run:

```powershell
python .\train_target_selection_ppo_v3.py --resume-path ..\checkpoints\target_selection_ppo_mlp_position_selfplay_latest_best.pt --iterations 40 --rollout-steps 3000 --ppo-epochs 4 --batch-size 256 --hidden-dim 64 --learning-rate 2.5e-4 --entropy-coef 0.02 --terminal-win-bonus 5.0 --reward-mode score_delta_cluster --opponent-mix bfs:0.15,rollout_gated:0.25,ppo:0.25,position_aware_ppo:0.35 --eval-opponents bfs rollout_gated ppo position_aware_ppo --selection-opponents bfs rollout_gated ppo position_aware_ppo --eval-seeds 0 1 2 3 4 --checkpoint-path .\checkpoints\target_selection_ppo_mlp_v3_latest.pt
```

Foreground league run:

```powershell
python .\league_train_v3.py --generations 5 --iterations 20
```

Background league run with controller logs:

```powershell
Start-Process -FilePath python -ArgumentList @('.\league_train_v3.py','--generations','5','--iterations','20') -WorkingDirectory 'C:\pythonlek\8sem\Reinforcement-Learning-\final_project\finalproject\part2\deeprl_v3' -RedirectStandardOutput 'C:\pythonlek\8sem\Reinforcement-Learning-\final_project\finalproject\part2\deeprl_v3\training_logs\league_loop_stdout.log' -RedirectStandardError 'C:\pythonlek\8sem\Reinforcement-Learning-\final_project\finalproject\part2\deeprl_v3\training_logs\league_loop_stderr.log' -WindowStyle Hidden
```

Short cluster-cap sweep, then automatic 20-iteration league restart:

```powershell
Start-Process -FilePath python -ArgumentList @('.\sweep_cluster_cap_v3.py') -WorkingDirectory 'C:\pythonlek\8sem\Reinforcement-Learning-\final_project\finalproject\part2\deeprl_v3' -RedirectStandardOutput 'C:\pythonlek\8sem\Reinforcement-Learning-\final_project\finalproject\part2\deeprl_v3\training_logs\sweep_then_league_stdout.log' -RedirectStandardError 'C:\pythonlek\8sem\Reinforcement-Learning-\final_project\finalproject\part2\deeprl_v3\training_logs\sweep_then_league_stderr.log' -WindowStyle Hidden
```

Watch current generation:

```powershell
Get-Content C:\pythonlek\8sem\Reinforcement-Learning-\final_project\finalproject\part2\deeprl_v3\training_logs\league_gen01_train.log -Wait -Tail 40
```

Watch league accept/reject summaries:

```powershell
Get-Content C:\pythonlek\8sem\Reinforcement-Learning-\final_project\finalproject\part2\deeprl_v3\training_logs\league_after_sweep_stdout.log -Wait -Tail 20
```

Watch the sweep controller:

```powershell
Get-Content C:\pythonlek\8sem\Reinforcement-Learning-\final_project\finalproject\part2\deeprl_v3\training_logs\sweep_then_league_stdout.log -Wait -Tail 20
```

## Current Manual League Restart

Gen1's final iteration-20 checkpoint was manually archived as:

```text
checkpoints/ppo_league_gen1.pt
```

This was chosen from replay behavior instead of the `_best` eval checkpoint.
The restarted gen2 run uses it both as the warm start and as a 25% opponent:

```text
bfs                 0.1125
rollout_gated       0.1875
ppo                 0.1875
position_aware_ppo  0.2625
ppo_league_gen1     0.2500
```

The active settings are:

```text
cluster_signal_max_abs=0.50
rollout_steps=5000
iterations=20
train_eval_seeds=0..4
selection_gate=3/5 wins vs position_aware_ppo and ppo_league_gen1, positive mean vs all eval opponents
gate_eval_seeds=0..9
acceptance_gate=6/10 wins and positive mean vs BFS/rollout_gated/PPO/position_aware_ppo
```

## Overnight League Attempt

After gen03 was abandoned, the next run is explicitly numbered gen04-gen06:

```powershell
Start-Process -FilePath python -ArgumentList @('.\league_train_v3.py','--start-generation','4','--generations','3','--iterations','20','--rollout-steps','5000','--cluster-signal-max-abs','0.50','--train-eval-seeds','0','1','2','3','4','--gate-eval-seeds','0','1','2','3','4','5','6','7','8','9','--league-weight','0.25','--league-recency-decay','0.75') -WorkingDirectory 'C:\pythonlek\8sem\Reinforcement-Learning-\final_project\finalproject\part2\deeprl_v3' -RedirectStandardOutput 'C:\pythonlek\8sem\Reinforcement-Learning-\final_project\finalproject\part2\deeprl_v3\training_logs\league_overnight_gen04_06_stdout.log' -RedirectStandardError 'C:\pythonlek\8sem\Reinforcement-Learning-\final_project\finalproject\part2\deeprl_v3\training_logs\league_overnight_gen04_06_stderr.log' -WindowStyle Hidden
```

Gen07 is queued to start after gen06 finishes, with a higher league pressure:

```text
league_weight=0.40
```

After gen06 finished, both the stale `0.25` watcher and the intended `0.40`
watcher launched gen07 at the same time. The mixed gen07 output/checkpoints were
archived with `_mixed_20260509_120413`, both duplicate processes were stopped,
and gen07 was restarted cleanly with only:

```text
league_weight=0.40
```

## Submission Candidate

The gen2 iteration-14 checkpoint was archived as:

```text
checkpoints/ppo_league_gen2_iter14.pt
```

It was packaged for upload as:

```text
part2/submission_agents/ppo_self_play_v1.zip
```

The zip contains top-level `agent.py`, `config.yaml`, and `model.pth`, plus a
duplicate `weights/model.pth` fallback.

## Success Criterion

Current target to beat:

```text
position-aware PPO:
BFS 0..9:           +9.00
rollout_gated 0..9: +6.70
```

Do not package v3 unless it beats this on 10-seed validation and does not
regress badly against the old PPO and position-aware PPO snapshots.
