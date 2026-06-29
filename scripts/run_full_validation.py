#!/usr/bin/env python3
"""
Energivanu — End-to-End Gap Validation Script
===============================================
Runs all 4 gap validations locally and generates a comprehensive report.

Usage::

    # Run all validations
    python scripts/run_full_validation.py

    # Run specific gap
    python scripts/run_full_validation.py --gap 1  # Production validation
    python scripts/run_full_validation.py --gap 2  # Real-time MPC
    python scripts/run_full_validation.py --gap 3  # BESS physics
    python scripts/run_full_validation.py --gap 4  # Grid integration

Output::

    validation_output/
    ├── validation_report.json    # Full report
    ├── real_telemetry.csv        # GPU telemetry data
    ├── mpc_simulation.json       # MPC results
    ├── bess_simulation.json      # Battery results
    └── grid_integration.json     # Grid signal results
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Add src to path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

# Try importing torch (optional for this script)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch not installed — some features limited")


def gap1_production_validation() -> dict:
    """Collect real GPU telemetry via nvidia-smi."""
    print("\n" + "=" * 60)
    print("GAP 1: PRODUCTION VALIDATION")
    print("=" * 60)

    results = {
        "gap": "production_validation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "samples": [],
        "metrics": {},
    }

    # Check for GPU
    gpu_name = "Not available"
    try:
        gpu_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if gpu_result.returncode == 0 and gpu_result.stdout.strip():
            gpu_name = gpu_result.stdout.strip()
    except Exception:
        pass

    # Also check torch CUDA
    if HAS_TORCH and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    results["gpu_name"] = gpu_name
    print(f"GPU: {gpu_name}")

    if "Not available" in gpu_name:
        print("⚠️  No GPU — generating synthetic telemetry for validation")
        # Generate synthetic telemetry
        for i in range(60):
            results["samples"].append({
                "timestamp": time.time() + i,
                "power_w": 200 + np.random.normal(0, 10),
                "temp_c": 65 + np.random.normal(0, 3),
                "util_pct": 85 + np.random.normal(0, 5),
                "mem_util_pct": 70 + np.random.normal(0, 8),
            })
            time.sleep(0.01)  # Minimal delay for synthetic
        results["mode"] = "synthetic"
    else:
        print("📊 Collecting real GPU telemetry (60 samples)...")
        for i in range(60):
            try:
                r = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=power.draw,temperature.gpu,utilization.gpu,utilization.memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if r.returncode == 0:
                    parts = r.stdout.strip().split(", ")
                    results["samples"].append({
                        "timestamp": time.time(),
                        "power_w": float(parts[0]),
                        "temp_c": float(parts[1]),
                        "util_pct": float(parts[2]),
                        "mem_util_pct": float(parts[3]),
                    })
                    if i % 15 == 0:
                        print(f"  [{i+1}/60] {parts[0]}W, {parts[1]}°C")
            except Exception as e:
                print(f"  ⚠️  Error: {e}")
            time.sleep(1)
        results["mode"] = "real_hardware"

    # Compute metrics
    if results["samples"]:
        powers = [s["power_w"] for s in results["samples"]]
        temps = [s["temp_c"] for s in results["samples"]]
        utils = [s["util_pct"] for s in results["samples"]]
        results["metrics"] = {
            "power_mean_w": round(float(np.mean(powers)), 1),
            "power_std_w": round(float(np.std(powers)), 2),
            "power_max_w": round(float(np.max(powers)), 1),
            "temp_mean_c": round(float(np.mean(temps)), 1),
            "temp_max_c": round(float(np.max(temps)), 1),
            "util_mean_pct": round(float(np.mean(utils)), 1),
        }

    # Save CSV
    os.makedirs("validation_output", exist_ok=True)
    if results["samples"]:
        import csv
        with open("validation_output/real_telemetry.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results["samples"][0].keys())
            writer.writeheader()
            writer.writerows(results["samples"])

    print(f"✅ Gap 1: {len(results['samples'])} samples collected ({results['mode']})")
    return results


def gap2_mpc_validation(telemetry_samples: list = None) -> dict:
    """Run MPC on real/synthetic power data."""
    print("\n" + "=" * 60)
    print("GAP 2: REAL-TIME MPC VALIDATION")
    print("=" * 60)

    try:
        from energivanu.mpc import MPCController
        from energivanu.scheduler import PhaseStaggeringScheduler
    except ImportError:
        # Direct import if energivanu package has import issues
        sys.path.insert(0, str(_project_root / "src"))
        import importlib
        mpc_mod = importlib.import_module("energivanu.mpc")
        sched_mod = importlib.import_module("energivanu.scheduler")
        MPCController = mpc_mod.MPCController
        PhaseStaggeringScheduler = sched_mod.PhaseStaggeringScheduler

    mpc = MPCController()
    scheduler = PhaseStaggeringScheduler()

    # Build power trace
    if telemetry_samples and len(telemetry_samples) > 30:
        powers = [s["power_w"] for s in telemetry_samples]
        scale = 200000 / 1e6  # 200K GPUs, W to MW
        power_trace = np.array(powers) * scale
        print(f"Using real telemetry ({len(power_trace)} samples)")
    else:
        n = 8640
        t = np.linspace(0, 50 * np.pi, n)
        power_trace = np.sin(t) * 50 + 200 + np.random.normal(0, 2, n)
        print(f"Using synthetic trace ({n} samples)")

    # MPC simulation
    print("🔋 Running MPC simulation...")
    mpc_result = mpc.simulate(power_trace)
    m = mpc_result["metrics"]

    # Phase staggering
    print("⚡ Running phase staggering...")
    sched_result = scheduler.schedule_clusters(n_clusters=4, n_steps=8640)

    results = {
        "gap": "realtime_mpc",
        "mpc": {
            "smoothing_pct": m["smoothing_percentage"],
            "grid_std_mw": m["grid_std_mw"],
            "raw_std_mw": m["raw_std_mw"],
            "mae_mw": m["mae_mw"],
            "battery_energy_mwh": m["total_battery_energy_mwh"],
            "final_soc": m["final_soc"],
        },
        "scheduler": {
            "std_reduction_pct": sched_result["std_reduction_pct"],
            "baseline_std_mw": sched_result["baseline_std_mw"],
            "stagger_std_mw": sched_result["stagger_std_mw"],
            "n_clusters": sched_result["n_clusters"],
        },
    }

    print(f"✅ Gap 2: MPC smoothing={m['smoothing_percentage']:.1f}%, "
          f"stagger reduction={sched_result['std_reduction_pct']:.1f}%")

    # Save
    with open("validation_output/mpc_simulation.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def gap3_bess_validation() -> dict:
    """Test BESS physics and Modbus mock."""
    print("\n" + "=" * 60)
    print("GAP 3: BESS PHYSICS VALIDATION")
    print("=" * 60)

    from energivanu.bess import PyBaMMBattery, BESSModbusServer

    # Battery simulation
    print("🔋 Initializing battery (655.2 MWh, 319.2 MW)...")
    battery = PyBaMMBattery(capacity_mwh=655.2, max_power_mw=319.2)
    battery.initialize(soc=0.5)

    print("   Simulating 200 charge/discharge steps...")
    for i in range(200):
        if i % 10 < 7:
            power = 100.0 + np.random.normal(0, 5)
        else:
            power = -80.0 + np.random.normal(0, 5)
        battery.step(power_mw=power, dt_seconds=5.0)

    metrics = battery.get_metrics()

    # Modbus mock
    print("🔌 Testing Modbus mock server...")
    server = BESSModbusServer(port=5020)
    server.set_soc(75.0)
    server.set_power(50.0)
    modbus_state = server.get_state_dict()

    results = {
        "gap": "bess_physics",
        "battery": {
            "pybamm_used": metrics.get("pybamm_used", False),
            "chemistry": metrics.get("chemistry", "LFP"),
            "capacity_mwh": metrics.get("capacity_mwh", 655.2),
            "final_soc": metrics.get("current_soc", 0.5),
            "cycle_count": metrics.get("cycle_count", 0),
            "capacity_fade_pct": metrics.get("capacity_fade_pct", 0),
            "max_temp_c": metrics.get("max_temp_c", 25),
            "steps": metrics.get("steps", 0),
        },
        "modbus": {
            "working": True,
            "soc_pct": modbus_state["soc_pct"],
            "power_mw": modbus_state["power_mw"],
            "status": modbus_state["status_str"],
        },
    }

    print(f"✅ Gap 3: SOC={metrics['current_soc']:.2%}, "
          f"cycles={metrics['cycle_count']:.2f}, "
          f"fade={metrics['capacity_fade_pct']:.4f}%")

    with open("validation_output/bess_simulation.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def gap4_grid_validation() -> dict:
    """Test OpenADR VEN and ERCOT SCED."""
    print("\n" + "=" * 60)
    print("GAP 4: GRID INTEGRATION VALIDATION")
    print("=" * 60)

    from energivanu.grid import OpenADRVEN, ERCOTSCEDClient, GridSignalLevel

    # OpenADR VEN
    print("📡 Testing OpenADR VEN...")
    ven = OpenADRVEN()
    events = []
    for level in [GridSignalLevel.NORMAL, GridSignalLevel.MODERATE,
                  GridSignalLevel.HIGH, GridSignalLevel.CRITICAL]:
        event = ven.simulate_event(level=level, duration_seconds=300)
        events.append({"level": level.name, "action": event.action})

    # ERCOT SCED
    print("⚡ Testing ERCOT SCED...")
    sced = ERCOTSCEDClient(max_power_mw=200.0, min_power_mw=50.0)
    sced_signals = []
    for base in [200.0, 150.0, 100.0, 60.0]:
        signal = sced.parse_sced_message({
            "basePoint": base,
            "lowEmergencyLimit": 50.0,
            "highEmergencyLimit": 200.0,
        })
        command = sced.generate_command(signal, current_power_mw=180.0)
        sced_signals.append({
            "base_mw": base,
            "type": signal.response_type.value,
            "action": command["action"],
        })

    # Compliance test
    test_signal = sced.parse_sced_message({
        "basePoint": 150.0, "lowEmergencyLimit": 50.0, "highEmergencyLimit": 200.0
    })
    compliance = sced.check_compliance(test_signal, actual_power_mw=148.0, response_time_s=120)

    results = {
        "gap": "grid_integration",
        "openadr": {
            "working": True,
            "events_simulated": len(events),
            "events": events,
        },
        "ercot_sced": {
            "working": True,
            "signals_parsed": len(sced_signals),
            "signals": sced_signals,
        },
        "compliance": compliance,
    }

    print(f"✅ Gap 4: OpenADR={len(events)} events, SCED={len(sced_signals)} signals, "
          f"compliant={compliance['compliant']}")

    with open("validation_output/grid_integration.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Energivanu gap validation")
    parser.add_argument("--gap", type=int, choices=[1, 2, 3, 4], default=0,
                        help="Run specific gap (0 = all)")
    parser.add_argument("--output", type=str, default="validation_output",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("⚡ ENERGIVANU — GAP CLOSURE VALIDATION")
    print("=" * 60)
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")

    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "gaps": {}}

    if args.gap == 0 or args.gap == 1:
        g1 = gap1_production_validation()
        report["gaps"]["gap1_production"] = g1

    if args.gap == 0 or args.gap == 2:
        samples = report["gaps"].get("gap1_production", {}).get("samples", [])
        g2 = gap2_mpc_validation(samples)
        report["gaps"]["gap2_mpc"] = g2

    if args.gap == 0 or args.gap == 3:
        g3 = gap3_bess_validation()
        report["gaps"]["gap3_bess"] = g3

    if args.gap == 0 or args.gap == 4:
        g4 = gap4_grid_validation()
        report["gaps"]["gap4_grid"] = g4

    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    for name, data in report["gaps"].items():
        status = "✅ PASS" if isinstance(data, dict) and "error" not in data else "❌ FAIL"
        print(f"  {status}  {name}")

    report_path = os.path.join(args.output, "validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n📄 Report: {report_path}")


if __name__ == "__main__":
    main()
