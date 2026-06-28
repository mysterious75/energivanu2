# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Export EnergivanuPEB Model to ONNX Format
===========================================
Loads a trained PyTorch checkpoint, exports to ONNX (opset 17),
and validates numerical accuracy between PyTorch and ONNX outputs.

Usage::

    # Export with default checkpoint
    python scripts/export_onnx.py

    # Export specific checkpoint
    python scripts/export_onnx.py --checkpoint models/checkpoints/commercial_best.pt

    # Export with custom output path
    python scripts/export_onnx.py --output models/onnx/energivanu.onnx

    # Validate only (skip export)
    python scripts/export_onnx.py --validate-only models/onnx/energivanu.onnx

Requirements::

    pip install onnx onnxruntime
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from energivanu.config import get_config
from energivanu.logging_config import get_logger, setup_logging, timed
from energivanu.model import EnergivanuPEB, load_model

logger = get_logger("export_onnx")


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------

def _build_validation_report(
    checkpoint_path: str,
    onnx_path: str,
    max_abs_diff: float,
    mean_abs_diff: float,
    max_rel_diff: float,
    power_passed: bool,
    signal_passed: bool,
    tolerance: float,
    export_time_s: float,
    onnx_size_mb: float,
) -> Dict[str, Any]:
    """Build a structured validation report."""
    return {
        "checkpoint": checkpoint_path,
        "onnx_model": onnx_path,
        "opset_version": 17,
        "tolerance": tolerance,
        "validation": {
            "power_output": {
                "max_abs_diff": round(max_abs_diff, 8),
                "mean_abs_diff": round(mean_abs_diff, 8),
                "max_rel_diff": round(max_rel_diff, 8),
                "passed": power_passed,
            },
            "signal_output": {
                "passed": signal_passed,
            },
            "overall_passed": power_passed and signal_passed,
        },
        "model_size_mb": round(onnx_size_mb, 2),
        "export_time_s": round(export_time_s, 3),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ---------------------------------------------------------------------------
# Core export logic
# ---------------------------------------------------------------------------

@timed("export_onnx.export")
def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    tolerance: float = 1e-5,
    opset_version: int = 17,
) -> Dict[str, Any]:
    """
    Export a trained EnergivanuPEB checkpoint to ONNX format.

    Steps:
      1. Load PyTorch checkpoint
      2. Create dummy input matching config dimensions
      3. Export with torch.onnx.export (opset 17)
      4. Validate ONNX output matches PyTorch output (tolerance 1e-5)
      5. Save .onnx file and validation report

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        output_path: Path to save the .onnx model.
        tolerance: Maximum absolute difference for validation (default 1e-5).
        opset_version: ONNX opset version (default 17).

    Returns:
        Validation report dictionary.
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        logger.error(
            "onnx and onnxruntime are required. Install with: "
            "pip install onnx onnxruntime"
        )
        raise

    # Load checkpoint to get config
    logger.info("loading checkpoint", extra={"path": checkpoint_path})
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt.get("config", {})

    # Build model
    model = EnergivanuPEB(
        n_features=model_cfg.get("n_features", 15),
        seq_len=model_cfg.get("seq_len", 30),
        pred_horizon=model_cfg.get("pred_horizon", 10),
        tcn_channels=model_cfg.get("tcn_channels", [32, 64, 128]),
        tcn_kernels=model_cfg.get("tcn_kernels", [5, 3, 3]),
        attention_heads=model_cfg.get("attention_heads", 8),
        attention_dim=model_cfg.get("attention_dim", 128),
        hidden_dims=model_cfg.get("hidden_dims", [256, 128]),
        n_signal_classes=model_cfg.get("n_signal_classes", 3),
        dropout=0.0,  # Disable dropout for export
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_features = model_cfg.get("n_features", 15)
    seq_len = model_cfg.get("seq_len", 30)

    # Create dummy input
    dummy_input = torch.randn(1, seq_len, n_features, dtype=torch.float32)

    # Get PyTorch output for validation
    with torch.no_grad():
        pt_power, pt_signal = model(dummy_input)
    pt_power_np = pt_power.numpy()
    pt_signal_np = pt_signal.numpy()

    # Export to ONNX
    logger.info(
        "exporting to ONNX",
        extra={"output": output_path, "opset": opset_version},
    )
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["power_prediction", "signal_logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "power_prediction": {0: "batch_size"},
            "signal_logits": {0: "batch_size"},
        },
        do_constant_folding=True,
    )
    export_time = time.time() - t0

    onnx_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(
        "ONNX export complete",
        extra={"size_mb": round(onnx_size_mb, 2), "time_s": round(export_time, 3)},
    )

    # Validate ONNX model
    logger.info("validating ONNX model")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model structure valid")

    # Run ONNX inference
    session = ort.InferenceSession(output_path)
    onnx_inputs = {"input": dummy_input.numpy()}
    onnx_outputs = session.run(None, onnx_inputs)

    onnx_power = onnx_outputs[0]
    onnx_signal = onnx_outputs[1]

    # Compare outputs
    power_diff = np.abs(pt_power_np - onnx_power)
    max_abs_diff = float(power_diff.max())
    mean_abs_diff = float(power_diff.mean())

    # Relative diff (avoid division by zero)
    rel_diff = power_diff / (np.abs(pt_power_np) + 1e-8)
    max_rel_diff = float(rel_diff.max())

    power_passed = max_abs_diff < tolerance
    signal_passed = np.argmax(pt_signal_np, axis=-1) == np.argmax(onnx_signal, axis=-1)

    logger.info(
        "validation results",
        extra={
            "power_max_abs_diff": max_abs_diff,
            "power_mean_abs_diff": mean_abs_diff,
            "power_passed": power_passed,
            "signal_passed": bool(signal_passed.all()),
        },
    )

    # Build and save report
    report = _build_validation_report(
        checkpoint_path=checkpoint_path,
        onnx_path=output_path,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        max_rel_diff=max_rel_diff,
        power_passed=power_passed,
        signal_passed=bool(signal_passed.all()),
        tolerance=tolerance,
        export_time_s=export_time,
        onnx_size_mb=onnx_size_mb,
    )

    report_path = Path(output_path).with_suffix(".validation.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("validation report saved", extra={"path": str(report_path)})

    return report


def validate_existing(onnx_path: str, checkpoint_path: str, tolerance: float = 1e-5) -> Dict[str, Any]:
    """Validate an existing ONNX model against its PyTorch source."""
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        logger.error("onnx and onnxruntime required: pip install onnx onnxruntime")
        raise

    # Load PyTorch model
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt.get("config", {})
    model = EnergivanuPEB(
        n_features=model_cfg.get("n_features", 15),
        seq_len=model_cfg.get("seq_len", 30),
        pred_horizon=model_cfg.get("pred_horizon", 10),
        tcn_channels=model_cfg.get("tcn_channels", [32, 64, 128]),
        tcn_kernels=model_cfg.get("tcn_kernels", [5, 3, 3]),
        attention_heads=model_cfg.get("attention_heads", 8),
        attention_dim=model_cfg.get("attention_dim", 128),
        hidden_dims=model_cfg.get("hidden_dims", [256, 128]),
        n_signal_classes=model_cfg.get("n_signal_classes", 3),
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_features = model_cfg.get("n_features", 15)
    seq_len = model_cfg.get("seq_len", 30)
    dummy_input = torch.randn(1, seq_len, n_features, dtype=torch.float32)

    with torch.no_grad():
        pt_power, pt_signal = model(dummy_input)

    session = ort.InferenceSession(onnx_path)
    onnx_outputs = session.run(None, {"input": dummy_input.numpy()})

    max_abs_diff = float(np.abs(pt_power.numpy() - onnx_outputs[0]).max())
    passed = max_abs_diff < tolerance

    logger.info(
        "validation complete",
        extra={"max_abs_diff": max_abs_diff, "passed": passed},
    )
    return {"max_abs_diff": max_abs_diff, "passed": passed}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for ONNX export."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Export EnergivanuPEB model to ONNX format with validation.",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="models/checkpoints/commercial_best.pt",
        help="Path to PyTorch checkpoint (default: models/checkpoints/commercial_best.pt)",
    )
    parser.add_argument(
        "--output", type=str,
        default="models/onnx/energivanu.onnx",
        help="Output path for ONNX model (default: models/onnx/energivanu.onnx)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-5,
        help="Validation tolerance (default: 1e-5)",
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--validate-only", type=str, default=None,
        help="Validate an existing ONNX model instead of exporting",
    )
    args = parser.parse_args()

    if args.validate_only:
        report = validate_existing(args.validate_only, args.checkpoint, args.tolerance)
        print(f"\nValidation: {'PASSED' if report['passed'] else 'FAILED'}")
        print(f"  Max abs diff: {report['max_abs_diff']:.2e}")
    else:
        report = export_to_onnx(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            tolerance=args.tolerance,
            opset_version=args.opset,
        )

        status = "PASSED" if report["validation"]["overall_passed"] else "FAILED"
        print(f"\n{'=' * 60}")
        print(f"ONNX EXPORT {status}")
        print(f"{'=' * 60}")
        print(f"  Model:       {args.output}")
        print(f"  Size:        {report['model_size_mb']:.2f} MB")
        print(f"  Opset:       {report['opset_version']}")
        print(f"  Tolerance:   {args.tolerance}")
        print(f"  Power diff:  {report['validation']['power_output']['max_abs_diff']:.2e}")
        print(f"  Signal OK:   {report['validation']['signal_output']['passed']}")
        print(f"  Time:        {report['export_time_s']:.3f}s")
        print(f"  Report:      {Path(args.output).with_suffix('.validation.json')}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
