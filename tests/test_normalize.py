import os

import numpy as np
import pytest

from normalize import STATS_PATH, load_stats, normalize, unnormalize

# Small hand-made stats: two dimensions with different ranges.
S = {"min": np.array([0.0, 10.0]), "max": np.array([100.0, 20.0])}


def test_min_max_map_to_minus_one_and_one():
    """normalize(min) should return -1 and normalize(max) should return 1, in every dimension."""
    assert np.allclose(normalize(S["min"], S), [-1, -1])
    assert np.allclose(normalize(S["max"], S), [1, 1])


def test_midpoint_maps_to_zero():
    """normalize of the halfway point between min and max should return 0."""
    mid = (S["min"] + S["max"]) / 2
    assert np.allclose(normalize(mid, S), [0, 0])


def test_round_trip():
    """unnormalize(normalize(x)) should return the original x."""
    x = np.array([[0.0, 10.0], [25.0, 12.5], [100.0, 20.0]])  # (T, 2)
    assert np.allclose(unnormalize(normalize(x, S), S), x)


def test_keeps_shape():
    """normalize and unnormalize on a (T, 2) array should return a (T, 2) array."""
    x = np.zeros((7, 2))
    assert normalize(x, S).shape == (7, 2)
    assert unnormalize(x, S).shape == (7, 2)


@pytest.mark.skipif(not os.path.exists(STATS_PATH), reason="norm_stats.json not generated")
def test_real_stats_shapes():
    """load_stats should return (2,) stats for state and action, and (16,) for the keypoints."""
    stats = load_stats()
    assert stats["observation.state"]["min"].shape == (2,)
    assert stats["observation.environment_state"]["min"].shape == (16,)
    assert stats["action"]["min"].shape == (2,)
