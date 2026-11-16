"""Min-max normalization of Push-T data to [-1, 1], and back.

What gets normalized (all in workspace pixels, 0-512):
    observation.state              (2,)   pusher (x, y)                  model input
    observation.environment_state  (16,)  8 block keypoints, (x, y) each  model input
    action                         (2,)   cursor target (x, y)           model output

Each dimension uses its own min/max from data/pusht_keypoints/norm_stats.json (written by inspect_data.py):
    normalize:    x_n = 2 * (x - min) / (max - min) - 1
    unnormalize:  x   = (x_n + 1) / 2 * (max - min) + min

The same stats are used in training and evaluation. Predicted actions come out in [-1, 1] and are
unnormalized back to pixels before env.step.
"""
import json

import numpy as np

STATS_PATH = 'data/pusht_keypoints/norm_stats.json'


def load_stats(path=STATS_PATH):
    """Read norm_stats.json into {name: {"min": array, "max": array}}"""
    with open(path) as f:
        raw = json.load(f)
    return {name: {"min": np.array(s["min"]), "max": np.array(s["max"])} for name, s in raw.items()}


def normalize(x, s):
    """Map x to [-1, 1] using one column's stats, e.g. stats["action"]"""
    return 2 * (x - s["min"]) / (s["max"] - s["min"]) - 1


def unnormalize(x_n, s):
    """Map x_n from [-1, 1] back to original units using one column's stats"""
    return (x_n + 1) / 2 * (s["max"] - s["min"]) + s["min"]


if __name__ == "__main__":
    from data_utils import col, load_table

    # Round trip over the whole dataset: normalize, unnormalize, compare to the original.
    stats = load_stats()
    table = load_table()
    for name, s in stats.items():
        x = col(table, name).astype(np.float64)
        x_n = normalize(x, s)
        back = unnormalize(x_n, s)
        print(f"{name}: range [{x_n.min():.3f}, {x_n.max():.3f}], round-trip max error {np.abs(back - x).max():.2e}")
