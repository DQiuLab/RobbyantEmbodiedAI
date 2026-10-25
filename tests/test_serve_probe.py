import numpy as np

from src import serve_probe


def test_check_action_accepts_finite_14_dim_chunk():
    assert serve_probe.check_action(np.zeros((25, 14))) is None


def test_check_action_rejects_bad_chunks():
    assert "empty" in serve_probe.check_action(np.zeros((0,)))
    assert "dim" in serve_probe.check_action(np.zeros((25, 7)))
    bad = np.zeros((25, 14)); bad[3, 2] = np.nan
    assert "non-finite" in serve_probe.check_action(bad)


def test_random_observation_has_three_cameras_and_state():
    obs = serve_probe.random_observation(np.random.default_rng(0))
    assert {k for k in obs if k.startswith("observation.images.")} == {
        f"observation.images.{c}" for c in serve_probe.CAMERAS}
    assert obs["observation.state"].shape == (14,)
