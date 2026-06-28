# SPDX-License-Identifier: AGPL-3.0-or-later
"""Energivanu CLI — quick commands for prediction and optimization."""

import argparse

import numpy as np


def cmd_predict(args):
    from .model import EnergivanuPEB
    model = EnergivanuPEB()
    print(f"Model created: {model.count_parameters():,} parameters")
    print("Note: Train on real H100 data for production use.")
    print("Run: python -m energivanu.train_real")


def cmd_optimize(args):
    from .mpc import MPCController
    mpc = MPCController()
    n = 8640
    trace = np.zeros(n)
    for i in range(n):
        pos = i % 10
        if pos == 9:
            trace[i] = 70.0
        else:
            trace[i] = 140.0
        trace[i] += np.random.normal(0, 2)

    result = mpc.simulate(trace)
    m = result["metrics"]
    print(f"Smoothing:       {m['smoothing_percentage']:.1f}%")
    print(f"Grid std:        {m['grid_std_mw']:.2f} MW  (raw: {m['raw_std_mw']:.2f} MW)")
    print(f"MAE to target:   {m['mae_mw']:.2f} MW")
    print(f"Battery energy:  {m['total_battery_energy_mwh']:.1f} MWh")
    print(f"Final SOC:       {m['final_soc']:.2%}")


def cmd_serve(args):
    try:
        import uvicorn

        from .api import app
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    except ImportError:
        print("Install API dependencies: pip install energivanu[api]")


def cmd_demo(args):
    print("=" * 60)
    print("ENERGIVANU — ML Power Prediction for AI Data Centers")
    print("=" * 60)

    from .model import EnergivanuPEB
    model = EnergivanuPEB()
    print(f"\nModel: {model.count_parameters():,} parameters")

    from .mpc import MPCController
    mpc = MPCController()
    trace = np.concatenate([
        np.full(864, 140.0),
        np.full(864, 70.0),
        np.full(864, 140.0),
        np.full(864, 70.0),
    ])
    trace += np.random.normal(0, 2, len(trace))
    result = mpc.simulate(trace)
    print(f"MPC Smoothing:   {result['metrics']['smoothing_percentage']:.1f}%")

    from .scheduler import PhaseStaggeringScheduler
    sched = PhaseStaggeringScheduler()
    sched_result = sched.schedule_clusters(4, n_steps=8640)
    print(f"Scheduler:       {sched_result['std_reduction_pct']:.1f}% std reduction")

    from .optimizer import PeakShavingOptimizer
    opt = PeakShavingOptimizer()
    daily = np.array([100 + 30 * np.sin(2 * np.pi * (h - 6) / 24) for h in range(24)])
    monthly = np.tile(daily, 30)
    peak_result = opt.simulate_month(monthly)
    print(f"Peak shaving:    {peak_result['peak_reduction_pct']:.1f}% reduction")
    print(f"Monthly savings: ${peak_result['total_monthly_demand_savings_usd']:,.0f}")

    print("\n" + "=" * 60)
    print("All components functional. Train on real H100 data for production.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        prog="energivanu", description="ML power prediction for AI data centers"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("predict", help="Check model status")
    sub.add_parser("optimize", help="Run battery optimization demo")
    sub.add_parser("demo", help="Run full system demo")

    serve_parser = sub.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    commands = {
        "predict": cmd_predict,
        "optimize": cmd_optimize,
        "serve": cmd_serve,
        "demo": cmd_demo,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
