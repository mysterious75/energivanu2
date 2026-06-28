# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Energivanu PEB System — Peak Shaving Optimizer
================================================
Minimizes monthly demand charges by shaving peak power consumption.
"""

from typing import Dict, List, Optional

import numpy as np

_DEFAULT_CONFIG = {
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


class PeakShavingOptimizer:
    """Hierarchical optimizer for peak demand shaving."""

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = _DEFAULT_CONFIG

        self.config = config
        self.pricing = config["pricing"]
        self.bat_cfg = config["battery"]

        self.demand_charge_kw = self.pricing["demand_charge_per_kw_month"]
        self.peak_hours = set(self.pricing["peak_hours"])
        self.offpeak_hours = set(self.pricing["offpeak_hours"])
        self.peak_rate = self.pricing["peak_rate_per_kwh"]
        self.offpeak_rate = self.pricing["offpeak_rate_per_kwh"]

        self.P_max = self.bat_cfg["total_power_mw"]
        self.E_max = self.bat_cfg["total_capacity_mwh"]

        self.window_15min = 900
        self.step_seconds = config["mpc"]["step_seconds"]
        self.window_steps = self.window_15min // self.step_seconds

        self.peak_15min_history: List[float] = []
        self.monthly_peak_mw = 0.0
        self.soc = 0.5

    def reset(self) -> None:
        self.peak_15min_history = []
        self.monthly_peak_mw = 0.0
        self.soc = 0.5

    def get_tou_period(self, hour: int) -> str:
        if hour in self.peak_hours:
            return "peak"
        elif hour in self.offpeak_hours:
            return "offpeak"
        return "shoulder"

    def get_tou_rate(self, hour: int) -> float:
        period = self.get_tou_period(hour)
        if period == "peak":
            return self.peak_rate
        elif period == "offpeak":
            return self.offpeak_rate
        return self.pricing["shoulder_rate_per_kwh"]

    def update_15min_average(self, current_power_mw: float) -> float:
        self.peak_15min_history.append(current_power_mw)
        if len(self.peak_15min_history) > self.window_steps:
            self.peak_15min_history = self.peak_15min_history[-self.window_steps:]
        avg = np.mean(self.peak_15min_history)
        if avg > self.monthly_peak_mw:
            self.monthly_peak_mw = avg
        return avg

    def optimize(
        self,
        current_power_mw: float,
        hour: int,
        battery_soc: float,
        target_15min_avg: Optional[float] = None,
    ) -> Dict:
        avg_15min = self.update_15min_average(current_power_mw)
        if target_15min_avg is None:
            target_15min_avg = self.monthly_peak_mw

        tou_period = self.get_tou_period(hour)
        tou_rate = self.get_tou_rate(hour)

        if avg_15min > self.monthly_peak_mw:
            self.monthly_peak_mw = avg_15min

        strategy = "hold"
        battery_action = 0.0

        if tou_period == "offpeak" and battery_soc < 0.9:
            charge_rate = min(self.P_max * 0.8, self.P_max * (0.9 - battery_soc) / 0.3)
            battery_action = charge_rate
            strategy = "offpeak_charge"
        elif tou_period == "peak" and battery_soc > 0.15:
            excess = max(0, avg_15min - target_15min_avg)
            base_discharge = self.P_max * 0.3
            peak_extra = excess * 1.2 if excess > 0 else 0
            discharge = min(base_discharge + peak_extra, self.P_max * 0.95)
            discharge = min(discharge, self.P_max * (battery_soc - 0.15) / 0.25)
            battery_action = -discharge
            strategy = "peak_shave" if excess > 0 else "peak_support"
        elif tou_period == "shoulder":
            if avg_15min > target_15min_avg * 0.90:
                discharge = min(self.P_max * 0.4, self.P_max * (battery_soc - 0.15) / 0.25)
                battery_action = -discharge
                strategy = "shoulder_shave"

        if battery_action >= 0:
            self.soc = battery_soc + (battery_action * 0.92 * self.step_seconds / 3600) / self.E_max
        else:
            self.soc = battery_soc + (battery_action / 0.92 * self.step_seconds / 3600) / self.E_max
        self.soc = float(np.clip(self.soc, 0.1, 0.9))

        grid_power = current_power_mw + battery_action
        demand_reduction_kw = max(0, (avg_15min - grid_power) * 1000)
        monthly_savings = demand_reduction_kw * self.demand_charge_kw / 1000

        return {
            "battery_action_mw": round(battery_action, 4),
            "grid_power_mw": round(grid_power, 4),
            "avg_15min_mw": round(avg_15min, 4),
            "monthly_peak_mw": round(self.monthly_peak_mw, 4),
            "soc": round(self.soc, 4),
            "tou_period": tou_period,
            "tou_rate": tou_rate,
            "strategy": strategy,
            "est_monthly_demand_savings_usd": round(monthly_savings, 2),
        }

    def simulate_month(self, hourly_power_trace: np.ndarray) -> Dict:
        self.reset()
        results = []
        for h, power in enumerate(hourly_power_trace):
            hour_of_day = h % 24
            result = self.optimize(power, hour_of_day, self.soc)
            results.append(result)

        total_savings = sum(r["est_monthly_demand_savings_usd"] for r in results)
        peak_before = float(np.max(hourly_power_trace))
        peak_after = self.monthly_peak_mw

        return {
            "total_monthly_demand_savings_usd": round(total_savings, 2),
            "peak_before_mw": round(peak_before, 2),
            "peak_after_mw": round(peak_after, 2),
            "peak_reduction_mw": round(peak_before - peak_after, 2),
            "peak_reduction_pct": round((peak_before - peak_after) / peak_before * 100, 2),
            "n_hours": len(hourly_power_trace),
        }

    def estimate_annual_savings(self, monthly_savings_usd: float) -> Dict:
        annual_demand = monthly_savings_usd * 12
        rate_diff = self.peak_rate - self.offpeak_rate
        daily_arbitrage_mwh = self.E_max * 0.5
        annual_arbitrage = daily_arbitrage_mwh * 365 * rate_diff * 1000
        return {
            "annual_demand_charge_savings": round(annual_demand, 2),
            "annual_tou_arbitrage_savings": round(annual_arbitrage, 2),
            "total_annual_savings": round(annual_demand + annual_arbitrage, 2),
        }
