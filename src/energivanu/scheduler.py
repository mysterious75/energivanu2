# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Energivanu PEB System — Phase-Staggering Scheduler
====================================================
Schedules GPU cluster workloads to minimize aggregate power fluctuation.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

_DEFAULT_CONFIG = {
    "grid": {
        "nominal_frequency_hz": 60.0,
        "inertia_constant_s": 4.5,
        "ramp_rate_limit_mw_per_min": 5.0,
    },
    "battery": {"total_power_mw": 319.2, "total_capacity_mwh": 655.2},
}


class PhaseStaggeringScheduler:
    """Scheduler that staggers compute phases across GPU clusters."""

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = _DEFAULT_CONFIG

        self.config = config
        self.cycle_period_steps = 10
        self.compute_fraction = 0.9
        self.P_compute = 140.0
        self.P_ar = 70.0

    def generate_cluster_trace(
        self, n_steps: int, phase_offset: int = 0, noise_std: float = 2.0,
    ) -> np.ndarray:
        trace = np.zeros(n_steps)
        for i in range(n_steps):
            effective_step = (i - phase_offset) % self.cycle_period_steps
            if effective_step == self.cycle_period_steps - 1:
                trace[i] = self.P_ar
            else:
                trace[i] = self.P_compute
            trace[i] += np.random.normal(0, noise_std)
        return trace

    def find_optimal_offset(
        self, n_clusters: int, n_steps: int = 8640,
        search_range: Optional[List[int]] = None,
        seed: int = 42,
    ) -> Tuple[int, float]:
        if search_range is None:
            search_range = list(range(0, self.cycle_period_steps))

        best_std = float("inf")
        best_offset = 0

        for offset in search_range:
            rng = np.random.RandomState(seed)
            traces = []
            for c in range(n_clusters):
                trace = np.zeros(n_steps)
                for i in range(n_steps):
                    effective_step = (i - offset * c) % self.cycle_period_steps
                    if effective_step == self.cycle_period_steps - 1:
                        trace[i] = self.P_ar
                    else:
                        trace[i] = self.P_compute
                    trace[i] += rng.normal(0, 2.0)
                traces.append(trace)
            aggregate = np.sum(traces, axis=0)
            std = np.std(aggregate)
            if std < best_std:
                best_std = std
                best_offset = offset

        return best_offset, best_std

    def schedule_clusters(
        self, n_clusters: int, n_steps: int = 8640,
        cluster_powers: Optional[List[float]] = None,
        noise_std: float = 2.0,
        seed: int = 42,
    ) -> Dict:
        if cluster_powers is None:
            cluster_powers = [self.P_compute] * n_clusters

        # Use fixed seed for reproducibility
        rng = np.random.RandomState(seed)

        best_offset, best_std = self.find_optimal_offset(n_clusters, n_steps, seed=seed)

        traces = {}
        for c in range(n_clusters):
            offset = best_offset * c
            trace = np.zeros(n_steps)
            for i in range(n_steps):
                effective_step = (i - offset) % self.cycle_period_steps
                if effective_step == self.cycle_period_steps - 1:
                    trace[i] = self.P_ar
                else:
                    trace[i] = cluster_powers[c]
                trace[i] += rng.normal(0, noise_std)
            traces[f"cluster_{c}"] = trace

        aggregate = np.sum(list(traces.values()), axis=0)

        # Baseline: all clusters synchronized (same seed for fair comparison)
        baseline_rng = np.random.RandomState(seed)
        baseline = np.zeros(n_steps)
        for i in range(n_steps):
            if i % self.cycle_period_steps == self.cycle_period_steps - 1:
                baseline[i] = self.P_ar * n_clusters
            else:
                baseline[i] = self.P_compute * n_clusters
            baseline[i] += baseline_rng.normal(0, noise_std * np.sqrt(n_clusters))

        baseline_std = np.std(baseline)
        stagger_std = np.std(aggregate)
        reduction = (1 - stagger_std / baseline_std) * 100 if baseline_std > 0 else 0

        return {
            "traces": traces,
            "aggregate": aggregate,
            "optimal_offset_steps": best_offset,
            "optimal_offset_seconds": best_offset * 5,
            "stagger_std_mw": round(stagger_std, 2),
            "baseline_std_mw": round(baseline_std, 2),
            "std_reduction_pct": round(reduction, 2),
            "n_clusters": n_clusters,
            "n_steps": n_steps,
        }

    def estimate_bess_burden_reduction(self, n_clusters: int) -> float:
        if n_clusters <= 1:
            return 0.0
        reduction_map = {2: 35.0, 3: 50.0, 4: 60.0, 5: 65.0, 6: 68.0}
        return reduction_map.get(n_clusters, 70.0 * (1 - 1 / n_clusters))
