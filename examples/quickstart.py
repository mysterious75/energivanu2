# Quick Start Example
# pip install energivanu

from energivanu import MPCController, PhaseStaggeringScheduler, PeakShavingOptimizer
import numpy as np

# Generate synthetic LLM training power trace (compute + all-reduce cycles)
n_steps = 8640
power_trace = np.zeros(n_steps)
for i in range(n_steps):
    pos = i % 10
    if pos == 9:
        power_trace[i] = 70.0  # all-reduce (low power)
    else:
        power_trace[i] = 140.0  # compute (high power)
    power_trace[i] += np.random.normal(0, 2)

print("=" * 60)
print("1. MPC Battery Optimization")
mpc = MPCController()
result = mpc.simulate(power_trace)
m = result["metrics"]
print(f"   Smoothing:       {m['smoothing_percentage']:.1f}%")
print(f"   Grid std:        {m['grid_std_mw']:.2f} MW (raw: {m['raw_std_mw']:.2f} MW)")
print(f"   Battery energy:  {m['total_battery_energy_mwh']:.1f} MWh")
print(f"   Final SOC:       {m['final_soc']:.2%}")

print()
print("2. Phase Staggering (4 clusters)")
scheduler = PhaseStaggeringScheduler()
schedule = scheduler.schedule_clusters(n_clusters=4, n_steps=8640)
print(f"   Optimal offset:  {schedule['optimal_offset_seconds']}s")
print(f"   Std reduction:   {schedule['std_reduction_pct']:.1f}%")

print()
print("3. Peak Shaving (monthly)")
optimizer = PeakShavingOptimizer()
daily_profile = np.array([100 + 30 * np.sin(2 * np.pi * (h - 6) / 24) for h in range(24)])
monthly_trace = np.tile(daily_profile, 30)
peak_result = optimizer.simulate_month(monthly_trace)
print(f"   Peak reduction:  {peak_result['peak_reduction_pct']:.1f}%")
print(f"   Monthly savings: ${peak_result['total_monthly_demand_savings_usd']:,.0f}")
annual = optimizer.estimate_annual_savings(peak_result["total_monthly_demand_savings_usd"])
print(f"   Annual savings:  ${annual['total_annual_savings']:,.0f}")

print()
print("4. Combined Effect")
# Staggered + battery = less BESS burden, lower peak, more savings
print(f"   Without staggering: 100% BESS burden")
print(f"   With staggering:    {100 - schedule['std_reduction_pct']:.0f}% BESS burden")
print(f"   Peak shaving saves: ${annual['total_annual_savings']:,.0f}/year")
