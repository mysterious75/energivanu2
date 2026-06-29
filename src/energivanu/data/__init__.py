# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Energivanu Data Package
========================
Data loading, processing, and dataset construction for the PEB model.

Components:

- :mod:`h100_processor` — Real H100 Data Processor (York University format)
- :mod:`alibaba_processor` — Alibaba GPU Trace 2020 data processor
- :mod:`validator` — Data quality validation
- :mod:`provenance` — Data lineage tracking
"""

# Lazy imports to avoid requiring torch at package import time
def __getattr__(name):
    if name in ("RealH100Dataset", "build_dataloaders", "create_sequences",
                "load_node_data", "scale_to_facility"):
        from .h100_processor import (
            RealH100Dataset, build_dataloaders, create_sequences,
            load_node_data, scale_to_facility,
        )
        return locals()[name]
    elif name == "AlibabaTraceProcessor":
        from .alibaba_processor import AlibabaTraceProcessor
        return AlibabaTraceProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "RealH100Dataset",
    "build_dataloaders",
    "create_sequences",
    "load_node_data",
    "scale_to_facility",
    "AlibabaTraceProcessor",
]
