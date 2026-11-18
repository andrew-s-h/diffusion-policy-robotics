import numpy as np
import torch
from torch.utils.data import Dataset

from data_utils import load_episode, load_table
from normalize import load_stats, normalize


def episode_arrays(e, stats):
    """Normalize one episode and join state + keypoints into observations.

    T is the number of frames in the episode (e.g. 161 for episode 0).

    Args:
        e (dict): one episode from load_episode, with "state" (T, 2),
            "keypoints" (T, 16), and "action" (T, 2) in pixels
        stats (dict): from load_stats; {column name: {"min": array, "max": array}}

    Returns:
        tuple[np.ndarray, np.ndarray]: obs (T, 18) and actions (T, 2), both in [-1, 1]
    """

    # robot pusher
    state = normalize(e["state"], stats["observation.state"])
    # t-block
    keypoints = normalize(e["keypoints"], stats["observation.environment_state"])
    # target the pusher moves toward
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
        obs_windows.append(obs_padded[t:t+n_obs]) #(2, 18)
        action_windows.append(actions_padded[t:t+n_actions]) # (16, 2)
    return np.stack(obs_windows), np.stack(action_windows)

if __name__ == "__main__":
    stats = load_stats()
    e = load_episode(load_table(), 0)
    obs, actions = episode_arrays(e, stats)
    print(obs.shape, actions.shape)
    print(obs.min(), obs.max())

    obs_padded, actions_padded = pad_episode(obs, actions)
    print(obs_padded.shape, actions_padded.shape)   # (162, 18) (176, 2)

    obs_windows, action_windows = make_windows(obs_padded, actions_padded, len(obs))
    print(obs_windows.shape, action_windows.shape)  # (161, 2, 18) (161, 16, 2)

