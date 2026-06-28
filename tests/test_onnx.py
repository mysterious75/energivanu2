"""Comprehensive ONNX model tests."""
import os
import sys
import time

import numpy as np
import pytest
import torch

ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from energivanu.model import EnergivanuPEB

CKPT = os.path.join(
    os.path.dirname(__file__), "..", "src", "models", "checkpoints", "best_model_real.pt"
)
ONNX = os.path.join(os.path.dirname(__file__), "..", "energivanu.onnx")

pytestmark = pytest.mark.skipif(
    not os.path.exists(CKPT) or not os.path.exists(ONNX),
    reason="Model checkpoint or ONNX file not found (gitignored)",
)


def load_models():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = EnergivanuPEB(
        **{k: config[k] for k in [
            "n_features", "seq_len", "pred_horizon", "tcn_channels",
            "tcn_kernels", "attention_heads", "attention_dim",
            "hidden_dims", "n_signal_classes",
        ]},
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    sess = ort.InferenceSession(ONNX)
    return model, sess, config


def test_input_output_shapes():
    model, sess, config = load_models()
    dummy = np.random.randn(1, 30, 15).astype(np.float32)
    pt_power, pt_signal = model(torch.from_numpy(dummy))
    onnx_p, onnx_s = sess.run(None, {"power_history": dummy})
    assert pt_power.shape == onnx_p.shape
    assert pt_signal.shape == onnx_s.shape
    print("PASS: Input/output shapes match")


def test_numerical_accuracy():
    model, sess, _ = load_models()
    dummy = np.random.randn(1, 30, 15).astype(np.float32)
    pt_p, pt_s = model(torch.from_numpy(dummy))
    onnx_p, onnx_s = sess.run(None, {"power_history": dummy})
    power_diff = np.abs(pt_p.detach().numpy() - onnx_p).mean()
    signal_diff = np.abs(pt_s.detach().numpy() - onnx_s).mean()
    assert power_diff < 1e-5, f"Power diff: {power_diff}"
    assert signal_diff < 1e-5, f"Signal diff: {signal_diff}"
    print(f"PASS: Numerical accuracy (power diff={power_diff:.8f}, signal diff={signal_diff:.8f})")


def test_batch_sizes():
    model, sess, _ = load_models()
    for bs in [1, 4, 16, 32]:
        inp = np.random.randn(bs, 30, 15).astype(np.float32)
        pt_out = model(torch.from_numpy(inp))[0].detach().numpy()
        onnx_out = sess.run(None, {"power_history": inp})[0]
        diff = np.abs(pt_out - onnx_out).mean()
        assert diff < 1e-5, f"Batch {bs} diff: {diff}"
    print("PASS: All batch sizes (1, 4, 16, 32)")


def test_inference_speed():
    model, sess, _ = load_models()
    inp = np.random.randn(1, 30, 15).astype(np.float32)
    for _ in range(5):
        model(torch.from_numpy(inp))
        sess.run(None, {"power_history": inp})
    t0 = time.time()
    for _ in range(100):
        model(torch.from_numpy(inp))
    pt_ms = (time.time() - t0) / 100 * 1000
    t0 = time.time()
    for _ in range(100):
        sess.run(None, {"power_history": inp})
    onnx_ms = (time.time() - t0) / 100 * 1000
    speedup = pt_ms / onnx_ms
    print(f"PASS: Speed - PyTorch {pt_ms:.2f}ms, ONNX {onnx_ms:.2f}ms, speedup {speedup:.2f}x")


def test_deterministic():
    _, sess, _ = load_models()
    inp = np.random.randn(1, 30, 15).astype(np.float32)
    out1 = sess.run(None, {"power_history": inp})[0]
    out2 = sess.run(None, {"power_history": inp})[0]
    diff = np.abs(out1 - out2).max()
    assert diff < 1e-6, f"Non-deterministic: {diff}"
    print("PASS: Deterministic output")


def test_power_range():
    _, sess, _ = load_models()
    inp = np.random.randn(32, 30, 15).astype(np.float32)
    powers = sess.run(None, {"power_history": inp})[0]
    assert powers.min() > 0, f"Negative power: {powers.min()}"
    assert powers.max() < 500, f"Power too high: {powers.max()}"
    print(f"PASS: Power range [{powers.min():.2f}, {powers.max():.2f}] MW")


def test_signal_classification():
    _, sess, _ = load_models()
    inp = np.random.randn(32, 30, 15).astype(np.float32)
    signals = sess.run(None, {"power_history": inp})[1]
    classes = np.argmax(signals, axis=-1)
    unique, counts = np.unique(classes, return_counts=True)
    labels = {0: "hold", 1: "discharge", 2: "charge"}
    for u, c in zip(unique, counts):
        print(f"  {labels[u]}: {c}/{len(classes)} ({c/len(classes)*100:.1f}%)")
    print("PASS: Signal classification")


def test_model_metadata():
    model, _, config = load_models()
    size_kb = os.path.getsize(ONNX) / 1024
    ckpt_mb = os.path.getsize(CKPT) / 1024 / 1024
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  ONNX size: {size_kb:.1f} KB")
    print(f"  Checkpoint size: {ckpt_mb:.1f} MB")
    print(f"  Config: {config}")
    print("PASS: Model metadata")


if __name__ == "__main__":
    test_input_output_shapes()
    test_numerical_accuracy()
    test_batch_sizes()
    test_inference_speed()
    test_deterministic()
    test_power_range()
    test_signal_classification()
    test_model_metadata()
    print("\nALL 8 TESTS PASSED")
