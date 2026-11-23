import numpy as np
import torch
from torch.utils.data import Dataset

from data_utils import col, load_episode, load_table
from normalize import load_stats, normalize


def episode_arrays(e, stats):
    """Normalize one episode and join state + keypoints into observations.

    T is the number of frames in the episode (e.g. 161 for episode 0).

    Args:
        e (dict): one episode from load_episode, with "state" (T, 2),
            "keypoints" (T, 16), and "action" (T, 2) in pixels
        stats (dict): from load_stats; {column name: {"min": array, "max": array}}

    Returns:
        tuple[np.ndarray, np.ndarray]: obs = state+ keypoints (T, 18) and 
            actions (T, 2), both in [-1, 1]
    """

    # robot pusher pos, 2 points
    state = normalize(e["state"], stats["observation.state"])
    # t-block pos, 16 points
    keypoints = normalize(e["keypoints"], stats["observation.environment_state"])
    # target the pusher moves toward, 2 points
    actions = normalize(e["action"], stats["action"])

    obs = np.concatenate([state, keypoints], axis=1)
    return obs, actions


def pad_episode(obs, actions, n_obs=2, n_actions=16):
    """Pad the ends of one episode so every frame can start a full window.

    The window at frame t uses obs t-1..t and actions t..t+15. At t = 0 there
    is no obs -1, so the first obs is repeated in front. Near the end there
    are fewer than 16 actions left, so the last action is repeated after.
    
    This prepares each episode before windows are cut. Last action repeated.

    Args:
        obs (np.ndarray): (T, 18) from episode_arrays
        actions (np.ndarray): (T, 2) from episode_arrays
        n_obs (int): observations per window
        n_actions (int): actions per window

    Returns:
        tuple[np.ndarray, np.ndarray]: obs_padded (T + n_obs - 1, 18) and
            actions_padded (T + n_actions - 1, 2)
    """
    # get a single copy of first frame: (1, 18)
    first = np.repeat(obs[:1], n_obs - 1, axis=0)

    # copy final action, 15 times: (15, 2)
    last = np.repeat(actions[-1:], n_actions - 1, axis=0)

    # first in front: (T, 18) -> (T+1, 18)
    obs_padded = np.concatenate([first, obs], axis=0)

    # last behind: (T, 2) -> (T+15, 2)
    actions_padded = np.concatenate([actions, last], axis=0)

    return obs_padded, actions_padded


def make_windows(obs_padded, actions_padded, T, n_obs=2, n_actions=16):
    """Cut one (observations, actions) training pair for every frame of an episode.

    Window t holds obs t-1..t and actions t..t+15. Because of the padding,
    these are just the slices [t:t+n_obs] and [t:t+n_actions].

    Args:
        obs_padded (np.ndarray): (T + n_obs - 1, 18) from pad_episode
        actions_padded (np.ndarray): (T + n_actions - 1, 2) from pad_episode
        T (int): number of real frames, len(obs) before padding
        n_obs (int): observations per window
        n_actions (int): actions per window

    Returns:
        tuple[np.ndarray, np.ndarray]: obs_windows (T, n_obs, 18) and
            action_windows (T, n_actions, 2)
    """
    obs_windows = []
    action_windows = []
    for t in range(T):
        # frame t-1, t from the slicing 
        obs_windows.append(obs_padded[t:t+n_obs]) # (2, 18)
        # actions t...t+15 from slicing actions_padded
        action_windows.append(actions_padded[t:t+n_actions]) # (16, 2)
    return np.stack(obs_windows), np.stack(action_windows)


class PushTWindows(Dataset):
    """Every training window from all 206 Push-T keypoint episodes.

    Each episode is normalized, padded, and cut into one window per frame,
    then all episodes are joined: 25,650 windows in total.

    Each item is a pair (obs, actions):
        obs (torch.Tensor): (n_obs, 18), frames t-1..t
        actions (torch.Tensor): (n_actions, 2), actions t..t+15

    Args:
        n_obs (int): observations per window
        n_actions (int): actions per window
    """

    def __init__(self, n_obs=2, n_actions=16):
        table = load_table()
        stats = load_stats()
        episodes = np.unique(col(table, "episode_index"))
    
        all_obs, all_actions = [], []

        for ep in episodes:
            obs, actions = episode_arrays(load_episode(table, ep), stats)
            obs_padded, actions_padded = pad_episode(obs, actions, n_obs, n_actions)
            obs_windows, action_windows = make_windows(obs_padded, actions_padded, len(obs), n_obs, n_actions)
            # obs_windows: (T, 2, 18), T = frames in this episode
            # 2 is from t-1, t frames
            all_obs.append(obs_windows)
            # action_windows: (T, 16, 2), T = frames in this episode
            all_actions.append(action_windows)
        # all episodes joined: (25650, 2, 18)
        self.obs = torch.tensor(np.concatenate(all_obs), dtype=torch.float32)
        # all episodes joined (25650, 16, 2)
        self.actions = torch.tensor(np.concatenate(all_actions), dtype=torch.float32)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, i):
        return self.obs[i], self.actions[i]


if __name__ == "__main__":
    stats = load_stats()
    e = load_episode(load_table(), 0)
    obs, actions = episode_arrays(e, stats)
    print(f"episode 0: obs {obs.shape}, actions {actions.shape}")
    print(f"obs range: {obs.min():.2f} to {obs.max():.2f}")

    obs_padded, actions_padded = pad_episode(obs, actions)
    print(f"padded: obs {obs_padded.shape}, actions {actions_padded.shape}")

    obs_windows, action_windows = make_windows(obs_padded, actions_padded, len(obs))
    print(f"windows: obs {obs_windows.shape}, actions {action_windows.shape}")

    ds = PushTWindows()
    o, a = ds[0]
    print(f"dataset: {len(ds)} windows, obs {tuple(o.shape)}, actions {tuple(a.shape)}")

