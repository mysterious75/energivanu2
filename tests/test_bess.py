"""Tests for the BESS battery simulation (PyBaMM/simplified)."""
from energivanu.bess.pybamm_battery import PyBaMMBattery, BatteryState


def test_battery_default_init():
    bat = PyBaMMBattery()
    assert bat.config.capacity_mwh == 655.2
    assert bat.config.max_power_mw == 319.2
    assert bat.config.chemistry == "LFP"


def test_battery_initialize():
    bat = PyBaMMBattery()
    bat.initialize(soc=0.5)
    assert bat._soc == 0.5


def test_battery_step_charge():
    bat = PyBaMMBattery()
    bat.initialize(soc=0.5)
    state = bat.step(power_mw=-50.0, dt_seconds=5.0)
    assert isinstance(state, BatteryState)
    assert state.soc > 0.5
    assert state.power_mw < 0


def test_battery_step_discharge():
    bat = PyBaMMBattery()
    bat.initialize(soc=0.5)
    state = bat.step(power_mw=50.0, dt_seconds=5.0)
    assert isinstance(state, BatteryState)
    assert state.soc < 0.5
    assert state.power_mw > 0


def test_battery_soc_stays_in_bounds():
    bat = PyBaMMBattery(soc_min=0.05, soc_max=0.95)
    bat.initialize(soc=0.5)
    for _ in range(100):
        state = bat.step(power_mw=319.2, dt_seconds=5.0)
    assert state.soc >= 0.0
    assert state.soc <= 1.0


def test_battery_cannot_over_discharge():
    bat = PyBaMMBattery(soc_min=0.05, soc_max=0.95)
    bat.initialize(soc=0.05)
    state = bat.step(power_mw=319.2, dt_seconds=60.0)
    assert state.power_mw <= 319.2


def test_battery_cannot_over_charge():
    bat = PyBaMMBattery(soc_min=0.05, soc_max=0.95)
    bat.initialize(soc=0.95)
    state = bat.step(power_mw=-319.2, dt_seconds=60.0)
    assert state.power_mw >= -319.2


def test_battery_power_clipped_to_limits():
    bat = PyBaMMBattery(max_power_mw=100.0)
    bat.initialize(soc=0.5)
    state = bat.step(power_mw=999.0, dt_seconds=5.0)
    assert abs(state.power_mw) <= 100.0


def test_battery_voltage_positive():
    bat = PyBaMMBattery()
    bat.initialize(soc=0.5)
    state = bat.step(power_mw=50.0, dt_seconds=5.0)
    assert state.voltage_v > 0


def test_battery_temperature_reasonable():
    bat = PyBaMMBattery()
    bat.initialize(soc=0.5, temperature_c=25.0)
    for _ in range(50):
        state = bat.step(power_mw=200.0, dt_seconds=5.0)
    assert 15.0 <= state.temperature_c <= 55.0


def test_battery_cycles_increase():
    bat = PyBaMMBattery(capacity_mwh=10.0, max_power_mw=5.0)
    bat.initialize(soc=0.5)
    for _ in range(100):
        bat.step(power_mw=5.0, dt_seconds=3600.0)
        bat.step(power_mw=-5.0, dt_seconds=3600.0)
    metrics = bat.get_metrics()
    assert metrics["cycle_count"] > 0.5


def test_battery_get_state():
    bat = PyBaMMBattery()
    bat.initialize(soc=0.5)
    state = bat.get_state()
    assert isinstance(state, BatteryState)
    assert state.soc == 0.5


def test_battery_get_history():
    bat = PyBaMMBattery()
    bat.initialize(soc=0.5)
    bat.step(50.0, 5.0)
    bat.step(-30.0, 5.0)
    history = bat.get_history()
    assert len(history) == 2


def test_battery_reset():
    bat = PyBaMMBattery()
    bat.initialize(soc=0.5)
    bat.step(50.0, 5.0)
    bat.reset(soc=0.8)
    assert bat._soc == 0.8
    assert len(bat._history) == 0


def test_battery_metrics():
    bat = PyBaMMBattery()
    metrics = bat.get_metrics()
    assert metrics["steps"] == 0
    bat.initialize(soc=0.5)
    bat.step(50.0, 5.0)
    metrics = bat.get_metrics()
    assert metrics["steps"] == 1
    assert "current_soc" in metrics


def test_battery_different_chemistry():
    bat_nmc = PyBaMMBattery(chemistry="NMC")
    bat_nmc.initialize(soc=0.5)
    state = bat_nmc.step(50.0, 5.0)
    assert state.soc < 0.5


def test_battery_large_power_step():
    bat = PyBaMMBattery(max_power_mw=1000.0, capacity_mwh=1000.0)
    bat.initialize(soc=0.5)
    state = bat.step(power_mw=800.0, dt_seconds=5.0)
    assert abs(state.power_mw) <= 1000.0
