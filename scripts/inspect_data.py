import glob
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "data/pusht_keypoints"

# --- load ---
paths = sorted(glob.glob(f"{ROOT}/data/chunk-*/*.parquet"))
print(f"{len(paths)} parquet file(s)")

table = pa.concat_tables([pq.read_table(p) for p in paths]).sort_by("index")
print(table.schema)
print("rows:", table.num_rows)


def col(name):
    """Return a column as numpy: (N,) for scalars, (N, d) for vector columns."""
    c = table.column(name).combine_chunks()
    t = c.type
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        return c.flatten().to_numpy(zero_copy_only=False).reshape(len(c), -1)
    return c.to_numpy(zero_copy_only=False)


# --- episode / frame indexing checks ---
ep = col("episode_index")
fi = col("frame_index")

eps, lengths = np.unique(ep, return_counts=True)
print(f"\nepisodes: {len(eps)}")
print(f"length min/mean/max: {lengths.min()} / {lengths.mean():.1f} / {lengths.max()}")

d = np.diff(ep)
assert np.isin(d, [0, 1]).all(), "episode_index is not contiguous"
assert fi[0] == 0, "first frame_index is not 0"
starts = np.flatnonzero(d == 1) + 1
assert (fi[starts] == 0).all(), "frame_index doesn't restart at episode boundaries"
assert (np.diff(fi)[d == 0] == 1).all(), "gap in frame_index within an episode"
print("episode/frame indexing OK")

# --- ranges + normalization stats ---
stats = {}
for name in ["observation.state", "observation.environment_state", "action"]:
    x = col(name).astype(np.float64)
    lo, hi = x.min(axis=0), x.max(axis=0)
    print(f"\n{name}: shape {x.shape}")
    print("  min:", np.round(lo, 1))
    print("  max:", np.round(hi, 1))
    if (hi == lo).any():
        print("  WARNING: constant dims:", np.flatnonzero(hi == lo))
    stats[name] = {"min": lo.tolist(), "max": hi.tolist()}

with open(f"{ROOT}/norm_stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"\nsaved {ROOT}/norm_stats.json")