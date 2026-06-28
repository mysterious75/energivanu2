# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Telemetry → Training Format Adapter
=====================================
Convert per-GPU telemetry CSV output (from
:class:`~energivanu.telemetry.nvidia_smi_collector.NvidiaSmiCollector`) into
the 15-feature column format expected by the Energivanu PEB model and
:func:`energivanu.data.create_sequences`.

The collector saves **per-GPU rows** (one row per GPU per timestamp):

    timestamp, unix_ts, gpu_id, power_w, temp_c, util_pct,
    mem_util_pct, sm_clock_mhz, mem_clock_mhz

The training pipeline expects **per-timestamp columns** (one row per
timestamp with 15 named features):

    facility_mw, power_roc, power_roc2, power_roll_mean, power_roll_std,
    gpu_avg_power_norm, gpu_max_power_norm, gpu_avg_temp_norm,
    gpu_max_temp_norm, gpu_avg_util_norm, gpu_avg_mem_util_norm,
    cpu_util_est_norm, hour_sin, hour_cos, is_allreduce

Usage::

    from energivanu.telemetry.format_adapter import FormatAdapter

    adapter = FormatAdapter()
    adapter.convert("data/collections/run_20240101/telemetry.csv")
    # → writes data/collections/run_20240101/training_features.csv

    # Or programmatic access
    features_df = adapter.convert_file("input.csv")
    adapter.save(features_df, "output.csv")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from ..config import get_config
from ..logging_config import get_logger, timed

logger = get_logger("telemetry")

# ---------------------------------------------------------------------------
# Constants (matching data.py and nvidia_smi_collector.py)
# ---------------------------------------------------------------------------

_GPU_TDP_W: float = 700.0      # H100 TDP for power normalization
_MAX_TEMP: float = 100.0       # Temperature normalization denominator
_NUM_FEATURES: int = 15
_ROLLING_WINDOW: int = 250     # Rolling stats window size

FEATURE_NAMES: List[str] = [
    "facility_mw",
    "power_roc",
    "power_roc2",
    "power_roll_mean",
    "power_roll_std",
    "gpu_avg_power_norm",
    "gpu_max_power_norm",
    "gpu_avg_temp_norm",
    "gpu_max_temp_norm",
    "gpu_avg_util_norm",
    "gpu_avg_mem_util_norm",
    "cpu_util_est_norm",
    "hour_sin",
    "hour_cos",
    "is_allreduce",
]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class FormatAdapter:
    """
    Convert per-GPU telemetry rows into 15-feature training columns.

    The adapter:
    1. Loads the collector CSV (per-GPU rows).
    2. Aggregates per timestamp (mean across GPUs for avg, max for max).
    3. Scales power to facility MW.
    4. Computes derivatives and rolling statistics.
    5. Adds time encoding and all-reduce heuristic.
    6. Outputs a DataFrame with exactly 15 named columns.

    Args:
        num_gpus_facility: Total GPUs in facility for MW scaling.
        gpus_per_node: GPUs per node for scaling.
        cpu_util_estimate: Default CPU utilization estimate (0–1)
            since nvidia-smi doesn't provide CPU data.
        rolling_window: Window size for rolling mean/std computation.
    """

    def __init__(
        self,
        num_gpus_facility: int = 200_000,
        gpus_per_node: int = 8,
        cpu_util_estimate: float = 0.4,
        rolling_window: int = _ROLLING_WINDOW,
    ):
        self.num_gpus_facility = num_gpus_facility
        self.gpus_per_node = gpus_per_node
        self.cpu_util_estimate = cpu_util_estimate
        self.rolling_window = rolling_window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @timed("format_adapter.convert")
    def convert(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Convert a collector CSV file to training feature format.

        Args:
            input_path: Path to the per-GPU telemetry CSV.
            output_path: Path for the output CSV.  If ``None``, writes
                ``training_features.csv`` in the same directory as the input.

        Returns:
            Path to the output CSV file.

        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If the input CSV has no valid data.
        """
        features_df = self.convert_file(input_path)

        if output_path is None:
            output_path = str(
                Path(input_path).parent / "training_features.csv"
            )

        self.save(features_df, output_path)
        return output_path

    @timed("format_adapter.convert_file")
    def convert_file(self, input_path: str) -> pd.DataFrame:
        """
        Load and convert a collector CSV file to training features.

        Args:
            input_path: Path to the per-GPU telemetry CSV.

        Returns:
            DataFrame with 15 feature columns and a ``timestamp`` column.

        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If the CSV has no valid data after aggregation.
        """
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        df = pd.read_csv(str(path))
        if df.empty:
            raise ValueError(f"Input CSV is empty: {path}")

        logger.info(
            "loaded collector CSV",
            extra={
                "path": str(path),
                "rows": len(df),
                "columns": list(df.columns),
            },
        )

        return self.convert_dataframe(df)

    @timed("format_adapter.convert_dataframe")
    def convert_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a raw telemetry DataFrame to 15-feature training format.

        The input DataFrame must have at least: ``timestamp``, ``gpu_id``,
        ``power_w``, ``temp_c``, ``util_pct``, ``mem_util_pct``.

        Args:
            df: Raw per-GPU telemetry DataFrame.

        Returns:
            DataFrame with 15 feature columns and a ``timestamp`` column.

        Raises:
            ValueError: If required columns are missing.
        """
        required_cols = {"timestamp", "power_w", "temp_c", "util_pct", "mem_util_pct"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Available: {list(df.columns)}"
            )

        # Step 1: Aggregate per timestamp
        agg = self._aggregate_per_timestamp(df)

        if agg.empty:
            raise ValueError("No data remaining after timestamp aggregation")

        # Step 2: Extract features
        features = self._extract_features(agg)

        logger.info(
            "format conversion complete",
            extra={
                "input_rows": len(df),
                "output_rows": len(features),
                "features": len(FEATURE_NAMES),
            },
        )

        return features

    @staticmethod
    def save(df: pd.DataFrame, output_path: str) -> None:
        """
        Save a features DataFrame to CSV.

        Args:
            df: Features DataFrame.
            output_path: Output file path.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(path), index=False)
        logger.info("features saved", extra={"path": str(path), "rows": len(df)})

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _aggregate_per_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate per-GPU rows into one row per timestamp.

        - ``power_w``: sum across GPUs → node power, then compute avg/max
        - ``temp_c``: mean and max across GPUs
        - ``util_pct``: mean across GPUs
        - ``mem_util_pct``: mean across GPUs
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

        if df.empty:
            return df

        agg_dict = {
            "power_w": ["sum", "mean", "max"],
            "temp_c": ["mean", "max"],
            "util_pct": ["mean"],
            "mem_util_pct": ["mean"],
        }

        # Only aggregate columns that exist
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

        grouped = df.groupby("timestamp").agg(agg_dict)

        # Flatten MultiIndex columns
        grouped.columns = [
            f"{col}_{func}" if func != "sum" else "node_power_w"
            if col == "power_w" else f"{col}_{func}"
            for col, func in grouped.columns
        ]

        # Rename for clarity
        rename_map = {
            "power_w_mean": "gpu_avg_power_w",
            "power_w_max": "gpu_max_power_w",
            "temp_c_mean": "gpu_avg_temp_c",
            "temp_c_max": "gpu_max_temp_c",
            "util_pct_mean": "gpu_avg_util_pct",
            "mem_util_pct_mean": "gpu_avg_mem_util_pct",
        }
        grouped = grouped.rename(columns=rename_map)

        return grouped.reset_index()

    def _extract_features(self, agg: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the 15 training features from aggregated metrics.

        Args:
            agg: Aggregated DataFrame with columns: ``timestamp``,
                ``node_power_w``, ``gpu_avg_power_w``, ``gpu_max_power_w``,
                ``gpu_avg_temp_c``, ``gpu_max_temp_c``, ``gpu_avg_util_pct``,
                ``gpu_avg_mem_util_pct``.

        Returns:
            DataFrame with 15 named feature columns and ``timestamp``.
        """
        n = len(agg)
        scale = self.num_gpus_facility / self.gpus_per_node / 1e6

        # Extract arrays
        node_power_w = agg["node_power_w"].values.astype(np.float64)
        gpu_avg_power = agg["gpu_avg_power_w"].values.astype(np.float64)
        gpu_max_power = agg["gpu_max_power_w"].values.astype(np.float64)
        gpu_avg_temp = agg["gpu_avg_temp_c"].values.astype(np.float64)
        gpu_max_temp = agg["gpu_max_temp_c"].values.astype(np.float64)
        gpu_avg_util = agg["gpu_avg_util_pct"].values.astype(np.float64)
        gpu_avg_mem_util = agg["gpu_avg_mem_util_pct"].values.astype(np.float64)

        # 0. Facility power (MW)
        facility_mw = node_power_w * scale

        # 1–2. Power derivatives
        power_roc = np.diff(facility_mw, prepend=facility_mw[0])
        power_roc2 = np.diff(power_roc, prepend=power_roc[0])

        # 3–4. Rolling statistics
        window = min(self.rolling_window, n)
        if window > 1:
            series = pd.Series(facility_mw)
            power_roll_mean = series.rolling(window, min_periods=1).mean().values
            power_roll_std = series.rolling(window, min_periods=1).std().fillna(0).values
        else:
            power_roll_mean = facility_mw.copy()
            power_roll_std = np.zeros(n)

        # 5–6. Normalized GPU power
        gpu_avg_power_norm = gpu_avg_power / _GPU_TDP_W
        gpu_max_power_norm = gpu_max_power / _GPU_TDP_W

        # 7–8. Normalized temperature
        gpu_avg_temp_norm = gpu_avg_temp / _MAX_TEMP
        gpu_max_temp_norm = gpu_max_temp / _MAX_TEMP

        # 9–10. Normalized utilization
        gpu_avg_util_norm = gpu_avg_util / 100.0
        gpu_avg_mem_util_norm = gpu_avg_mem_util / 100.0

        # 11. CPU utilization estimate
        cpu_util_est_norm = np.full(n, self.cpu_util_estimate)

        # 12–13. Cyclical time encoding
        timestamps = pd.to_datetime(agg["timestamp"])
        hours = timestamps.dt.hour + timestamps.dt.minute / 60.0
        hour_sin = np.sin(2 * np.pi * hours.values / 24)
        hour_cos = np.cos(2 * np.pi * hours.values / 24)

        # 14. All-reduce heuristic
        is_allreduce = (
            (gpu_avg_util > 80) & (gpu_avg_mem_util < 30)
        ).astype(np.float64)

        # Assemble
        features = pd.DataFrame({
            "facility_mw": facility_mw,
            "power_roc": power_roc,
            "power_roc2": power_roc2,
            "power_roll_mean": power_roll_mean,
            "power_roll_std": power_roll_std,
            "gpu_avg_power_norm": gpu_avg_power_norm,
            "gpu_max_power_norm": gpu_max_power_norm,
            "gpu_avg_temp_norm": gpu_avg_temp_norm,
            "gpu_max_temp_norm": gpu_max_temp_norm,
            "gpu_avg_util_norm": gpu_avg_util_norm,
            "gpu_avg_mem_util_norm": gpu_avg_mem_util_norm,
            "cpu_util_est_norm": cpu_util_est_norm,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "is_allreduce": is_allreduce,
            "timestamp": agg["timestamp"].values,
        })

        # Final safety: replace any remaining NaN/inf
        for col in FEATURE_NAMES:
            features[col] = np.nan_to_num(features[col], nan=0.0, posinf=0.0, neginf=0.0)

        return features


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def create_adapter_from_config(config: Any = None) -> FormatAdapter:
    """
    Build a :class:`FormatAdapter` from the global configuration.

    Args:
        config: An :class:`EnergivanuConfig` instance.  If ``None``, loads
            the singleton config via :func:`get_config`.

    Returns:
        Configured adapter instance.
    """
    if config is None:
        config = get_config()

    return FormatAdapter(
        num_gpus_facility=config.training.num_gpus_target,
        gpus_per_node=config.training.gpus_per_node,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="energivanu-format",
        description="Convert telemetry CSV to training feature format",
    )
    parser.add_argument(
        "input",
        help="Path to the collector telemetry CSV file",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path (default: <input_dir>/training_features.csv)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int, default=200_000,
        help="Total facility GPUs for MW scaling (default: 200000)",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int, default=8,
        help="GPUs per node (default: 8)",
    )
    parser.add_argument(
        "--cpu-util",
        type=float, default=0.4,
        help="CPU utilization estimate, 0-1 (default: 0.4)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entry point for format conversion.

    Args:
        argv: Command-line arguments.  ``None`` uses ``sys.argv[1:]``.

    Returns:
        Exit code (0 for success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    adapter = FormatAdapter(
        num_gpus_facility=args.num_gpus,
        gpus_per_node=args.gpus_per_node,
        cpu_util_estimate=args.cpu_util,
    )

    try:
        output_path = adapter.convert(args.input, args.output)
        print(f"✅ Features saved to {output_path}")
        return 0
    except Exception as exc:
        print(f"❌ Conversion failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
