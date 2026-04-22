# Project Instructions

- For the Collector final project, prototype agents in `part2/agent_sketch.py`.
- Use `part2/agent_explore.ipynb` only for visualization and environment inspection.
- Use `part2/train_agent.py` for training loops and checkpoint saving.
- The final submitted agent must live in `src/agents/agent/agent.py`.
- Do not edit `src/agents/baseline/` or `src/agents/random/`.
- Keep `Agent.load(self) -> None`; `compete.py` calls `agent.load()` with no arguments.
- Prefer NumPy and PyTorch only. Do not use RL libraries like Stable Baselines, RLlib, or PufferLib for submitted code.
- Test final agents with `python src/compete/compete.py src/agents/agent src/agents/baseline`.

- Whenever proposing, implementing, or testing changes to the current Collector agent, also update `final_project/finalproject/part2/history.md` with the attempted change, motivation, evaluation/result if available, and decision/next step.

- At the start of every new sesion, read through history.md to see what we attempted to implement/experiment with last session. 

- When i ask you to change something, or say something about preferance in terms of code/answer, make an small note for yourself in best_practise.md 
