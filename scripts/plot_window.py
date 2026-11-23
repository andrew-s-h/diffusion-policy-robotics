"""Phase 2 check: do a window's 16 actions continue on from the pusher's path?

Draws one training window from PushTWindows in pixels: the pusher at
frames t-1 and t, and the 16 action targets t..t+15. If the windows line
up, the actions start near the pusher and keep going the same way.

Run from the repo root:
    python scripts/plot_window.py 80
"""
import sys

import matplotlib.pyplot as plt

from dataset import PushTWindows
from normalize import load_stats, unnormalize


if __name__ == "__main__":
    # Window number from the command line; default to 80 (middle of episode 0).
    i = int(sys.argv[1]) if len(sys.argv) > 1 else 80

    stats = load_stats()
    obs, actions = PushTWindows()[i]

    # Back to pixels. The first 2 numbers of each obs row are the pusher.
    pusher = unnormalize(obs[:, :2].numpy(), stats["observation.state"])   # (2, 2)
    targets = unnormalize(actions.numpy(), stats["action"])                # (16, 2)

    plt.plot(pusher[:, 0], pusher[:, 1], "o-", label="pusher (t-1, t)")
    plt.plot(targets[:, 0], targets[:, 1], ".-", label="actions (t..t+15)")
    plt.xlim(0, 512)
    plt.ylim(0, 512)
    plt.gca().set_aspect("equal")
    plt.title(f"window {i}")
    plt.legend()
    plt.show()
