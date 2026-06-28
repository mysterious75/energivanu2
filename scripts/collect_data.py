#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Energivanu Data Collection CLI
================================
Command-line script for collecting GPU telemetry data.

Wraps :class:`~energivanu.telemetry.data_collector.DataCollector` with an
argparse interface providing four commands:

- ``quick``     — 5-minute rapid sanity check
- ``standard``  — 1-hour typical training data collection
- ``marathon``  — 8+ hour production data gathering
- ``custom``    — user-specified duration

Usage::

    # Quick 5-minute test
    python scripts/collect_data.py quick

    # Standard 1-hour collection
    python scripts/collect_data.py standard

    # Marathon with custom output
    python scripts/collect_data.py marathon -o data/marathon_run

    # Custom 30-minute collection at 2-second intervals
    python scripts/collect_data.py custom --duration 0.5 --interval 2.0

    # Simulation mode (no real GPU needed)
    python scripts/collect_data.py quick --simulation
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="collect_data",
        description="Energivanu GPU telemetry data collection tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s quick                         5-minute quick test
  %(prog)s standard                      1-hour standard collection
  %(prog)s marathon -o data/run1         8-hour marathon with custom output
  %(prog)s custom -d 0.25 -i 2.0        15-minute collection, 2s interval
  %(prog)s quick --simulation            Simulated mode (no GPU needed)
        """,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        help="Collection mode",
    )

    # ------------------------------------------------------------------
    # Shared arguments helper
    # ------------------------------------------------------------------
    def add_common_args(p: argparse.ArgumentParser) -> None:
        """Add arguments common to all collection commands."""
        p.add_argument(
            "--output-dir", "-o",
            default="data/collections",
            metavar="DIR",
            help="Output directory (default: data/collections)",
        )
        p.add_argument(
            "--interval", "-i",
            type=float,
            default=1.0,
            metavar="SEC",
            help="Collection interval in seconds (default: 1.0)",
        )
        p.add_argument(
            "--simulation", "-s",
            action="store_true",
            default=False,
            help="Force simulation mode (no real GPU required)",
        )
        p.add_argument(
            "--rate",
            type=float,
            default=0.12,
            metavar="$/KWH",
            help="Electricity rate in $/kWh (default: 0.12)",
        )

    # ------------------------------------------------------------------
    # Subcommands
    # ------------------------------------------------------------------

    # quick — 5 minutes
    quick_p = subparsers.add_parser(
        "quick",
        help="5-minute quick collection (sanity check / debug)",
        description="Collect GPU telemetry for 5 minutes. "
                    "Useful for verifying setup or quick debugging.",
    )
    add_common_args(quick_p)

    # standard — 1 hour
    standard_p = subparsers.add_parser(
        "standard",
        help="1-hour standard collection (typical training data)",
        description="Collect GPU telemetry for 1 hour. "
                    "Good baseline for training data gathering.",
    )
    add_common_args(standard_p)

    # marathon — 8+ hours
    marathon_p = subparsers.add_parser(
        "marathon",
        help="8+ hour marathon collection (production data gathering)",
        description="Collect GPU telemetry for 8+ hours. "
                    "For production-grade data collection runs.",
    )
    add_common_args(marathon_p)
    marathon_p.add_argument(
        "--duration", "-d",
        type=float,
        default=8.0,
        metavar="HOURS",
        help="Duration in hours (default: 8.0)",
    )

    # custom — user-defined
    custom_p = subparsers.add_parser(
        "custom",
        help="Custom duration collection",
        description="Collect GPU telemetry for a user-specified duration.",
    )
    add_common_args(custom_p)
    custom_p.add_argument(
        "--duration", "-d",
        type=float,
        required=True,
        metavar="HOURS",
        help="Duration in hours (required)",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entry point.

    Args:
        argv: Command-line arguments.  ``None`` uses ``sys.argv[1:]``.

    Returns:
        Exit code (0 = success, 1 = error, 130 = interrupted).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    # Lazy import to avoid slow startup on --help
    from energivanu.telemetry.data_collector import CollectionMode, DataCollector

    mode = CollectionMode(args.command)
    duration = getattr(args, "duration", None)

    collector = DataCollector(
        mode=mode,
        duration_hours=duration,
        output_dir=args.output_dir,
        collection_interval_s=args.interval,
        simulation_mode=args.simulation,
        electricity_rate=args.rate,
    )

    try:
        result = collector.run()

        print()
        print("=" * 60)
        print("  Collection Complete")
        print("=" * 60)
        print(f"  Mode:       {result.mode}")
        print(f"  Duration:   {result.duration_actual_hours:.2f} h "
              f"(requested: {result.duration_requested_hours:.2f} h)")
        print(f"  Samples:    {result.total_samples}")
        print(f"  Energy:     {result.energy_kwh:.4f} kWh")
        print(f"  Cost:       ${result.cost_usd:.4f}")
        print(f"  Emissions:  {result.emissions_kg:.4f} kg CO₂")
        print(f"  Errors:     {result.errors}")
        print(f"  Sim mode:   {result.simulation_mode}")
        print()
        print(f"  CSV output: {result.output_csv}")
        if result.output_db:
            print(f"  DB output:  {result.output_db}")
        print("=" * 60)

        # Offer to convert to training format
        print()
        print("To convert to training features:")
        print(f"  python -m energivanu.telemetry.format_adapter {result.output_csv}")

        return 0

    except KeyboardInterrupt:
        print("\n⏹  Collection interrupted by user")
        return 130

    except Exception as exc:
        print(f"\n❌ Collection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
