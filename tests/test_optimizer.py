"""Tests for the PeakShavingOptimizer."""
import numpy as np

from energivanu.optimizer import PeakShavingOptimizer


def test_optimizer_default_init():
    opt = PeakShavingOptimizer()
    assert opt.demand_charge_kw == 15.0
    assert opt.soc == 0.5


def test_get_tou_period():
    opt = PeakShavingOptimizer()
    assert opt.get_tou_period(14) == "peak"
    assert opt.get_tou_period(0) == "offpeak"
    assert opt.get_tou_period(10) == "shoulder"


def test_get_tou_rate():
    opt = PeakShavingOptimizer()
    assert opt.get_tou_rate(14) == 0.12
    assert opt.get_tou_rate(0) == 0.05


def test_optimize_offpeak_charge():
    opt = PeakShavingOptimizer()
    result = opt.optimize(current_power_mw=150.0, hour=3, battery_soc=0.3)
    assert result["battery_action_mw"] > 0
    assert result["strategy"] == "offpeak_charge"


def test_optimize_peak_shave():
    opt = PeakShavingOptimizer()
    result = opt.optimize(current_power_mw=250.0, hour=15, battery_soc=0.8)
    assert result["battery_action_mw"] < 0
    assert "peak" in result["strategy"]


def test_optimize_shoulder_action():
    opt = PeakShavingOptimizer()
    for _ in range(200):
        opt.optimize(150.0, 10, 0.5)
    result = opt.optimize(current_power_mw=300.0, hour=10, battery_soc=0.8)
    assert result["tou_period"] == "shoulder"


def test_optimize_near_full_soc_no_charge():
    opt = PeakShavingOptimizer()
    result = opt.optimize(current_power_mw=150.0, hour=3, battery_soc=0.95)
    assert result["strategy"] == "hold"


def test_optimize_near_empty_soc_no_discharge():
    opt = PeakShavingOptimizer()
    result = opt.optimize(current_power_mw=250.0, hour=15, battery_soc=0.05)
    assert result["strategy"] == "hold"


def test_simulate_month():
    opt = PeakShavingOptimizer()
    hourly = np.array([200 + 50 * (i % 12) for i in range(720)])
    result = opt.simulate_month(hourly)
    assert "peak_before_mw" in result
    assert "peak_after_mw" in result
    assert "peak_reduction_pct" in result
    assert result["n_hours"] == 720


def test_simulate_month_reduces_peak():
    opt = PeakShavingOptimizer()
    hourly = np.array([200 + 100 * (i % 12) for i in range(720)])
    result = opt.simulate_month(hourly)
    assert result["peak_after_mw"] <= result["peak_before_mw"]


def test_estimate_annual_savings():
    opt = PeakShavingOptimizer()
    result = opt.estimate_annual_savings(monthly_savings_usd=50000)
    assert result["annual_demand_charge_savings"] == 600000
    assert result["total_annual_savings"] > result["annual_demand_charge_savings"]


def test_reset():
    opt = PeakShavingOptimizer()
    opt.optimize(250.0, 15, 0.8)
    opt.reset()
    assert opt.soc == 0.5
    assert opt.monthly_peak_mw == 0.0


def test_15min_rolling_average():
    config = {
        "mpc": {"step_seconds": 5},
        "pricing": {
            "demand_charge_per_kw_month": 15.0,
            "peak_rate_per_kwh": 0.12,
            "offpeak_rate_per_kwh": 0.05,
            "shoulder_rate_per_kwh": 0.08,
            "peak_hours": [14, 15, 16, 17, 18],
            "offpeak_hours": [0, 1, 2, 3, 4, 5, 6],
        },
        "battery": {"total_power_mw": 319.2, "total_capacity_mwh": 655.2},
    }
    opt = PeakShavingOptimizer(config)
    for i in range(200):
        opt.update_15min_average(float(150 + i))
    assert len(opt.peak_15min_history) <= opt.window_steps
