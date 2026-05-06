from pathlib import Path
import sys

PART2_ROOT = Path(__file__).resolve().parents[2]
if str(PART2_ROOT) not in sys.path:
    sys.path.insert(0, str(PART2_ROOT))

from deterministic_agents.bfs_agent import Agent
