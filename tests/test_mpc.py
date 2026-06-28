import numpy as np

from energivanu.mpc import MPCController


def test_mpc_default_init():
    mpc = MPCController()
    assert mpc.P_max == 319.2
    assert mpc.E_max == 655.2
    assert mpc.soc == 0.5


def test_mpc_optimize_returns_tuple():
    mpc = MPCController()
    history = [140.0]
    u, info = mpc.optimize(140.0, history)
    assert isinstance(u, float)
    assert "battery_action_mw" in info
    assert "grid_power_mw" in info
    assert "soc" in info


def test_mpc_soc_within_bounds():
    mpc = MPCController()
    mpc.reset(soc=0.5)
    history = []
    for i in range(100):
        power = 140.0 + np.random.normal(0, 5)
        history.append(power)
        _, info = mpc.optimize(power, history)
    assert 0.0 <= mpc.soc <= 1.0


def test_mpc_simulate():
    mpc = MPCController()
    trace = np.zeros(100)
    for i in range(100):
        trace[i] = 140.0 if i % 10 != 9 else 70.0
        trace[i] += np.random.normal(0, 1)
    result = mpc.simulate(trace)
    assert "metrics" in result
    assert "smoothing_percentage" in result["metrics"]
    assert result["metrics"]["n_steps"] == 100


def test_mpc_reset():
    mpc = MPCController()
    mpc.reset(soc=0.8)
    assert mpc.soc == 0.8
    assert mpc.step_count == 0
