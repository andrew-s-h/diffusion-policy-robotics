"""Replay recorded demo actions in gym-pusht and compare the simulated pusher path to the recorded one.

Run from the repo root:
    python scripts/replay_episode.py          # episodes 0, 1, 2
    python scripts/replay_episode.py 5 42     # specific episodes
"""
import sys

import gymnasium as gym
import gym_pusht  # registers gym_pusht/PushT-v0
import numpy as np

from data_utils import load_episode, load_table

# Observations come back as a dict with "agent_pos" and "environment_state" (block keypoints).
env = gym.make("gym_pusht/PushT-v0", obs_type="environment_state_agent_pos")
table = load_table()

# Episodes from the command line, or 0, 1, 2 by default.
for ep in [int(x) for x in sys.argv[1:]] or [0, 1, 2]:
    e = load_episode(table, ep)
    S, A = e["state"], e["action"]  # (T, 2) pusher positions, (T, 2) cursor targets

    # Simulator state: [agent_x, agent_y, block_x, block_y, block_angle].
    # Pusher starts at the demo's first position. Block pose is arbitrary
    env.reset(options={"reset_to_state": [*S[0], 300, 300, 0]})

    # env.step(a) returns (obs, reward, terminated, truncated, info).
    # obs is a dict: {"agent_pos": (2,), "environment_state": (16,)}.
    sim = np.array([env.step(a)[0]["agent_pos"] for a in A])

    err = np.linalg.norm(sim[:-1] - S[1:], axis=1)
    print(f"episode {ep}: mean {err.mean():.2f} px, max {err.max():.2f} px")
