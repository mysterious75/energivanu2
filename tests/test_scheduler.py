"""Tests for the PhaseStaggeringScheduler."""
import numpy as np

from energivanu.scheduler import PhaseStaggeringScheduler


def test_scheduler_default_init():
    sched = PhaseStaggeringScheduler()
    assert sched.cycle_period_steps == 10
    assert sched.compute_fraction == 0.9


def test_generate_cluster_trace():
    sched = PhaseStaggeringScheduler()
    trace = sched.generate_cluster_trace(n_steps=100, phase_offset=0)
    assert len(trace) == 100
    assert np.all(trace > 0)


def test_generate_cluster_trace_with_offset():
    sched = PhaseStaggeringScheduler()
    t0 = sched.generate_cluster_trace(100, phase_offset=0)
    t1 = sched.generate_cluster_trace(100, phase_offset=5)
    assert not np.allclose(t0, t1)


def test_find_optimal_offset():
    sched = PhaseStaggeringScheduler()
    offset, std = sched.find_optimal_offset(n_clusters=4)
    assert 0 <= offset < 10
    assert std > 0


def test_schedule_clusters_return_structure():
    sched = PhaseStaggeringScheduler()
    result = sched.schedule_clusters(n_clusters=4)
    assert "traces" in result
    assert "aggregate" in result
    assert "optimal_offset_steps" in result
    assert "std_reduction_pct" in result
    assert len(result["traces"]) == 4


def test_schedule_clusters_reduces_variance():
    sched = PhaseStaggeringScheduler()
    result = sched.schedule_clusters(n_clusters=4, n_steps=1000)
    assert result["std_reduction_pct"] > 0


def test_schedule_different_cluster_counts():
    sched = PhaseStaggeringScheduler()
    for n in [1, 2, 3, 4, 5]:
        result = sched.schedule_clusters(n_clusters=n, n_steps=500)
        assert result["n_clusters"] == n
        if n > 1:
            assert result["std_reduction_pct"] > 0


def test_schedule_with_custom_powers():
    sched = PhaseStaggeringScheduler()
    result = sched.schedule_clusters(n_clusters=3, cluster_powers=[100.0, 200.0, 300.0])
    assert result["n_clusters"] == 3


def test_estimate_bess_burden_reduction():
    sched = PhaseStaggeringScheduler()
    assert sched.estimate_bess_burden_reduction(1) == 0.0
    assert sched.estimate_bess_burden_reduction(4) > 0


def test_schedule_reproducible():
    sched = PhaseStaggeringScheduler()
    r1 = sched.schedule_clusters(n_clusters=4, seed=42)
    r2 = sched.schedule_clusters(n_clusters=4, seed=42)
    assert r1["std_reduction_pct"] == r2["std_reduction_pct"]
