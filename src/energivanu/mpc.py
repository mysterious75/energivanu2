# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Energivanu PEB System — MPC Controller (v4)
=============================================
Trajectory-based MPC for Tesla Megapack battery dispatch.
Four optimization strategies: proportional, constant trajectory, two-phase, swing.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

_DEFAULT_CONFIG = {
    "mpc": {
        "horizon_steps": 12,
        "step_seconds": 5,
        "soc_min": 0.05,
        "soc_max": 0.95,
        "efficiency": 0.92,
        "max_power_mw": 319.2,
        "total_capacity_mwh": 655.2,
        "Q": 100.0,
        "R": 0.01,
        "S": 0.1,
        "grid_target_mw": 200.0,
    },
    "grid": {
        "nominal_frequency_hz": 60.0,
        "inertia_constant_s": 4.5,
        "ramp_rate_limit_mw_per_min": 5.0,
        "facility_current_mw": 200.0,
        "frequency_deadband_hz": 0.02,
    },
}


class MPCController:
    """MPC for Tesla Megapack battery dispatch with trajectory optimization."""

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = _DEFAULT_CONFIG

        self.config = config
        mpc = config["mpc"]
        grid = config["grid"]

        self.N = mpc["horizon_steps"]
        self.dt = mpc["step_seconds"]
        self.soc_min = mpc["soc_min"]
        self.soc_max = mpc["soc_max"]
        self.eta = mpc["efficiency"]
        self.P_max = mpc["max_power_mw"]
        self.E_max = mpc["total_capacity_mwh"]
        self.Q = mpc.get("Q", 100.0)
        self.R = mpc.get("R", 0.01)
        self.S = mpc.get("S", 0.1)

        self.grid_target = mpc.get("grid_target_mw", grid.get("facility_current_mw", 200.0))
        self.f0 = grid["nominal_frequency_hz"]
        self.H = grid["inertia_constant_s"]
        self.grid_ramp_limit = grid["ramp_rate_limit_mw_per_min"]
        self.freq_deadband = grid.get("frequency_deadband_hz", 0.02)

        self.prev_u = 0.0
        self.prev_grid = self.grid_target
        self.step_count = 0
        self.soc = 0.5

    def reset(self, soc: float = 0.5):
        self.soc = soc
        self.prev_u = 0.0
        self.prev_grid = self.grid_target
        self.step_count = 0

    def _forecast(self, current_power: float, history: List[float]) -> np.ndarray:
        forecast = np.full(self.N, current_power)
        if len(history) >= 20:
            recent = np.array(history[-20:])
            trend = np.polyfit(range(len(recent)), recent, 1)[0]
            for k in range(self.N):
                forecast[k] = current_power + trend * (k + 1)
        return forecast

    def _simulate_traj(self, traj: np.ndarray, forecast: np.ndarray, soc0: float) -> float:
        soc = soc0
        cost = 0.0
        prev_u = self.prev_u

        for k in range(len(traj)):
            u = traj[k]
            P = forecast[k] + u
            dev = P - self.grid_target
            cost += self.Q * dev ** 2
            cost += self.R * u ** 2
            cost += self.S * (u - prev_u) ** 2

            if u >= 0:
                soc += (u * self.eta * self.dt / 3600) / self.E_max
            else:
                soc += (u / self.eta * self.dt / 3600) / self.E_max

            if soc < self.soc_min - 0.1 or soc > self.soc_max + 0.1:
                cost += 5000.0
            prev_u = u

        return cost

    def optimize(self, current_power: float, history: List[float],
                 target_power: Optional[float] = None) -> Tuple[float, Dict]:
        """Find optimal battery action."""
        if target_power is None:
            target_power = self.grid_target

        forecast = self._forecast(current_power, history)
        best_cost = float("inf")
        best_u = 0.0

        for gain in [0.3, 0.5, 0.7, 0.9, 1.0]:
            deviation = current_power - target_power
            u = -gain * deviation
            u = float(np.clip(u, -self.P_max, self.P_max))
            if u > 0 and self.soc >= self.soc_max:
                u = 0.0
            elif u < 0 and self.soc <= self.soc_min:
                u = 0.0
            cost = abs(u) * self.R
            if cost < best_cost:
                best_cost = cost
                best_u = u

        for u in np.linspace(-self.P_max * 0.8, self.P_max * 0.8, 11):
            traj = np.full(self.N, u)
            cost = self._simulate_traj(traj, forecast, self.soc)
            if cost < best_cost:
                best_cost = cost
                best_u = u

        for u1 in np.linspace(-self.P_max * 0.5, self.P_max * 0.5, 7):
            for u2 in np.linspace(-self.P_max * 0.5, self.P_max * 0.5, 7):
                traj = np.concatenate([
                    np.full(self.N // 2, u1),
                    np.full(self.N - self.N // 2, u2),
                ])
                cost = self._simulate_traj(traj, forecast, self.soc)
                if cost < best_cost:
                    best_cost = cost
                    best_u = u1

        if len(history) >= 10:
            cyc = 10
            pos = self.step_count % cyc
            if pos == 9:
                swing_u = min(self.P_max, (target_power - current_power) * 0.8)
            elif pos == 8:
                swing_u = -min(self.P_max * 0.5, (current_power - target_power) * 0.5)
            elif pos == 0:
                swing_u = min(self.P_max * 0.3, (target_power - current_power) * 0.3)
            else:
                swing_u = -min(self.P_max * 0.3, (current_power - target_power) * 0.3)
            swing_u = float(np.clip(swing_u, -self.P_max, self.P_max))
            deviation = current_power + swing_u - target_power
            cost_swing = abs(swing_u) * self.R + self.Q * deviation ** 2
            if cost_swing < best_cost:
                best_cost = cost_swing
                best_u = swing_u

        if best_u > 0 and self.soc >= self.soc_max:
            best_u = 0.0
        elif best_u < 0 and self.soc <= self.soc_min:
            best_u = 0.0

        best_u = float(np.clip(best_u, -self.P_max, self.P_max))

        if best_u >= 0:
            self.soc += (best_u * self.eta * self.dt / 3600) / self.E_max
        else:
            self.soc += (best_u / self.eta * self.dt / 3600) / self.E_max
        self.soc = float(np.clip(self.soc, self.soc_min, self.soc_max))

        grid_power = current_power + best_u
        dP = (self.grid_target - grid_power) / 1000.0
        df = self.f0 * dP / (2 * self.H)

        info = {
            "battery_action_mw": round(best_u, 4),
            "grid_power_mw": round(grid_power, 4),
            "soc": round(self.soc, 4),
            "cycle_pos": self.step_count % 10,
            "target_power": round(target_power, 4),
            "freq_deviation_hz": round(df, 6),
        }

        self.prev_u = best_u
        self.prev_grid = grid_power
        self.step_count += 1
        return best_u, info

    def simulate(self, power_trace: np.ndarray,
                 target_power: Optional[float] = None) -> Dict:
        """Run MPC simulation over a power trace."""
        self.reset(soc=0.5)
        if target_power is None:
            target_power = float(np.mean(power_trace))

        batt, grids, socs, freqs = [], [], [], []
        history: List[float] = []

        for P_it in power_trace:
            history.append(float(P_it))
            _, info = self.optimize(float(P_it), history, target_power)
            batt.append(info["battery_action_mw"])
            grids.append(info["grid_power_mw"])
            socs.append(info["soc"])
            freqs.append(info["freq_deviation_hz"])

        grids = np.array(grids)
        raw = np.array(power_trace)

        grid_std = float(np.std(grids))
        raw_std = float(np.std(raw))
        smoothing = (1 - grid_std / raw_std) * 100 if raw_std > 0 else 0.0

        mae = float(np.mean(np.abs(grids - target_power)))
        rmse = float(np.sqrt(np.mean((grids - target_power) ** 2)))
        max_dev = float(np.max(np.abs(grids - target_power)))

        freq_viol = int(np.sum(np.abs(np.array(freqs)) > self.freq_deadband))

        ramp_limit = self.grid_ramp_limit
        ramp_viol = 0
        for i in range(1, len(grids)):
            ramp = abs(grids[i] - grids[i - 1]) / (self.dt / 60)
            if ramp > ramp_limit:
                ramp_viol += 1

        batt_energy = float(np.sum(np.abs(batt)) * self.dt / 3600)

        metrics = {
            "smoothing_percentage": round(smoothing, 2),
            "grid_std_mw": round(grid_std, 4),
            "raw_std_mw": round(raw_std, 4),
            "mae_mw": round(mae, 4),
            "rmse_mw": round(rmse, 4),
            "max_deviation_mw": round(max_dev, 4),
            "target_power_mw": round(target_power, 4),
            "freq_violations_count": freq_viol,
            "freq_violations_pct": round(freq_viol / len(power_trace) * 100, 2),
            "ramp_violations_count": ramp_viol,
            "ramp_violations_pct": round(ramp_viol / len(power_trace) * 100, 2),
            "total_battery_energy_mwh": round(batt_energy, 2),
            "final_soc": round(self.soc, 4),
            "n_steps": len(power_trace),
        }

        return {
            "metrics": metrics,
            "battery_actions": batt,
            "grid_powers": grids.tolist(),
            "soc_profile": socs,
            "frequency_deviations": freqs,
        }

    def simulate_with_staggering(self, cluster_a_power: np.ndarray,
                                 stagger_seconds: int = 3) -> Dict:
        """Compare with and without phase-staggering."""
        stagger_steps = max(1, int(stagger_seconds / self.dt))
        cluster_b = np.roll(cluster_a_power, stagger_steps)
        aggregate = cluster_a_power + cluster_b

        res_no = self.simulate(cluster_a_power * 2)
        res_yes = self.simulate(aggregate)

        e_no = res_no["metrics"]["total_battery_energy_mwh"]
        e_yes = res_yes["metrics"]["total_battery_energy_mwh"]
        reduction = ((e_no - e_yes) / e_no * 100) if e_no > 0 else 0.0

        return {
            "no_stagger": res_no,
            "with_stagger": res_yes,
            "reduction_in_burden_pct": round(reduction, 2),
        }
