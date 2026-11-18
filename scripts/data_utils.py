import glob

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "data/pusht_keypoints"

def load_table(root=ROOT):
    """Read every parquet file into one Arrow table"""
    paths = sorted(glob.glob(f"{root}/data/chunk-*/*.parquet"))
    return pa.concat_tables([pq.read_table(p) for p in paths]).sort_by("index")


def col(table, name):
    """Column as numpy: (N, ) for scalars, (N, d) for list columns"""
    c = table.column(name).combine_chunks()
    t = c.type
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        return c.flatten().to_numpy(zero_copy_only=False).reshape(len(c), -1)
    return c.to_numpy(zero_copy_only=False)

def load_episode(table, episode):
    """Returns all frames of one episode sorted by time as a dictionary of numpy arrays.
    T is the number of frames in the episode (e.g. 161 for episode 0).
    Args:
        table (pa.Table): from load_table
        episode (int): episode index, 0 to 205
    Returns:
        dict: in pixels, not normalized
            "state" (T, 2): pusher (x, y), from observation.state
            "keypoints" (T, 16): 8 (x, y) corners of the T block,
                from observation.environment_state
            "action" (T, 2): target (x, y) the pusher moves toward
            "success" (T,) bool: True once the block covers the goal (95%+)
    """
    mask = col(table, "episode_index") == episode
    if not mask.any():
        raise ValueError(f"no episode {episode}")
    return {
        "state": col(table, "observation.state")[mask].astype(np.float64),
        "keypoints": col(table, "observation.environment_state")[mask].astype(np.float64),
        "action": col(table, "action")[mask].astype(np.float64),
        "success": col(table, "next.success")[mask]
    }

