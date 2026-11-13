"""Phase 0 sanity check: does the pusher (state) trail the cursor (action)?

Run from the repo root:
    python scripts/plot_episode.py 0
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

from data_utils import load_episode, load_table


def lag_table(state, action, max_k=5):
    """Average distance between the cursor at time t and the pusher at time t+k.

    state, action: (T, 2) arrays of (x, y) pixel positions.
    Returns {k: mean distance in px} for k = 0..max_k.
    The k with the smallest distance is the pusher's delay behind the cursor.
    """
    T = len(state)
    errors = {}
    for k in range(max_k + 1):
        # Pair cursor at t with pusher at t+k.
        # Both slices have length T-k: action[i] lines up with state[i+k].
        a = action[: T - k]
        s = state[k:]
        # Per-frame straight-line distance: sqrt(dx^2 + dy^2), one value per row.
        dist = np.linalg.norm(a - s, axis=1)
        # Average over frames -> one number for this k.
        errors[k] = dist.mean()
    return errors


def plot_episode(e, title=""):
    """Three panels: x over time, y over time, and the 2D path on the table."""
    s, a = e["state"], e["action"]
    t = np.arange(len(s))  # frame numbers 0..T-1

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panels 1 and 2: each coordinate over time.
    # Expect the action curve to move first and the state curve to follow a few frames later.
    for i, name in enumerate(["x", "y"]):
        axes[i].plot(t, s[:, i], label="state (pusher)")
        axes[i].plot(t, a[:, i], label="action (cursor)", alpha=0.7)
        axes[i].set_xlabel("frame")
        axes[i].set_ylabel(f"{name} (px)")
        axes[i].legend()

    # Panel 3: both paths drawn on the 512x512 table, viewed from above.
    # Expect them to mostly overlap, with the cursor path jumpier.
    axes[2].plot(s[:, 0], s[:, 1], label="state")
    axes[2].plot(a[:, 0], a[:, 1], label="action", alpha=0.7)
    axes[2].set_xlim(0, 512)
    axes[2].set_ylim(0, 512)
    axes[2].set_aspect("equal")  # 1 px in x looks the same as 1 px in y
    axes[2].set_title("2D path")
    axes[2].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Episode number from the command line; default to episode 0.
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    e = load_episode(load_table(), ep)

    for k, err in lag_table(e["state"], e["action"]).items():
        print(f"k={k}: mean |a_t - s_(t+k)| = {err:.1f} px")

    plot_episode(e, title=f"episode {ep}")