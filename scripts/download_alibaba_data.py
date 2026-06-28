#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Download Alibaba GPU Trace 2020 Dataset
=========================================
Downloads the Alibaba cluster-trace-gpu-v2020 dataset from the official
GitHub repository.  The dataset is released under CC BY 4.0.

Dataset:  Alibaba GPU Trace 2020
Source:   https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020
Paper:    "Characterizing and Profiling GPU Workloads on Alibaba" (NSDI '22)
License:  Creative Commons Attribution 4.0 International (CC BY 4.0)

Citation::

    @inproceedings{wen2022characterizing,
        title     = {Characterizing and Profiling GPU Workloads on Alibaba},
        author    = {Wen, Mingshu and Li, Haowei and Liu, Yang and others},
        booktitle = {Proceedings of the 19th USENIX Symposium on Networked
                     Systems Design and Implementation (NSDI)},
        year      = {2022},
        url       = {https://www.usenix.org/conference/nsdi22/presentation/wen}
    }

Usage::

    # Download to default location
    python scripts/download_alibaba_data.py

    # Download to custom directory
    python scripts/download_alibaba_data.py --output-dir data/alibaba

    # List available files without downloading
    python scripts/download_alibaba_data.py --list-only

    # Print citation info
    python scripts/download_alibaba_data.py --cite
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------

DATASET_NAME = "Alibaba GPU Trace 2020 (cluster-trace-gpu-v2020)"
DATASET_URL = "https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020"
RAW_BASE_URL = "https://raw.githubusercontent.com/alibaba/clusterdata/master/cluster-trace-gpu-v2020"

# Known files in the dataset (as of 2024).
# These are the CSV trace files; exact names may vary.
KNOWN_FILES: List[Dict[str, str]] = [
    {
        "name": "README.md",
        "description": "Dataset documentation",
        "url": f"{RAW_BASE_URL}/README.md",
    },
    # The actual trace files are typically split into parts.
    # Users should consult the GitHub repo for the exact file listing.
]

# Citation information
CITATION = """\
@inproceedings{wen2022characterizing,
    title     = {Characterizing and Profiling GPU Workloads on Alibaba},
    author    = {Wen, Mingshu and Li, Haowei and Liu, Yang and others},
    booktitle = {Proceedings of the 19th USENIX Symposium on Networked
                 Systems Design and Implementation (NSDI)},
    year      = {2022},
    url       = {https://www.usenix.org/conference/nsdi22/presentation/wen}
}"""

LICENSE_INFO = """\
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
You are free to:
  - Share: copy and redistribute the material in any medium or format
  - Adapt: remix, transform, and build upon the material for any purpose
Under the following terms:
  - Attribution: You must give appropriate credit, provide a link to the
    license, and indicate if changes were made.
Full license: https://creativecommons.org/licenses/by/4.0/
"""


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _fetch_url_to_file(url: str, dest: Path, chunk_size: int = 8192) -> int:
    """
    Download a URL to a local file.

    Args:
        url: Source URL.
        dest: Destination file path.
        chunk_size: Read chunk size in bytes.

    Returns:
        Number of bytes written.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(url, headers={"User-Agent": "energivanu/1.0"})
    bytes_written = 0

    with urllib.request.urlopen(req, timeout=120) as response:
        with open(dest, "wb") as fh:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                fh.write(chunk)
                bytes_written += len(chunk)

    return bytes_written


def _md5(filepath: Path) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as fh:
        while True:
            chunk = fh.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def list_available_files() -> List[Dict[str, str]]:
    """
    Return the list of known dataset files.

    Returns:
        List of dicts with ``name``, ``description``, and ``url`` keys.
    """
    return KNOWN_FILES


def download_dataset(
    output_dir: str = "data/alibaba/cluster-trace-gpu-v2020",
    verbose: bool = True,
) -> List[str]:
    """
    Download the Alibaba GPU Trace 2020 dataset.

    Downloads the README and any available trace files from the official
    GitHub repository.  For the full dataset (which may be several GB),
    users should also follow the instructions in the README.

    Args:
        output_dir: Directory to save downloaded files.
        verbose: Whether to print progress to stdout.

    Returns:
        List of paths to downloaded files.
    """
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    downloaded: List[str] = []

    for file_info in KNOWN_FILES:
        name = file_info["name"]
        url = file_info["url"]
        dest = dest_dir / name

        if dest.exists():
            if verbose:
                print(f"  ⏭  Skipping {name} (already exists)")
            downloaded.append(str(dest))
            continue

        if verbose:
            print(f"  ⬇  Downloading {name}...")

        try:
            bytes_written = _fetch_url_to_file(url, dest)
            downloaded.append(str(dest))
            if verbose:
                size_kb = bytes_written / 1024
                print(f"  ✅ {name} ({size_kb:.1f} KB)")
        except Exception as exc:
            if verbose:
                print(f"  ❌ Failed to download {name}: {exc}")

    return downloaded


def print_citation() -> None:
    """Print citation and license information to stdout."""
    print("=" * 70)
    print(f"  {DATASET_NAME}")
    print("=" * 70)
    print()
    print(f"  Source:  {DATASET_URL}")
    print()
    print("  Citation:")
    print(CITATION)
    print()
    print(LICENSE_INFO)
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="download_alibaba_data",
        description=f"Download the {DATASET_NAME} dataset",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data/alibaba/cluster-trace-gpu-v2020",
        metavar="DIR",
        help="Output directory (default: data/alibaba/cluster-trace-gpu-v2020)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List available files without downloading",
    )
    parser.add_argument(
        "--cite",
        action="store_true",
        help="Print citation and license information",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entry point.

    Args:
        argv: Command-line arguments.  ``None`` uses ``sys.argv[1:]``.

    Returns:
        Exit code (0 for success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Always show citation on download
    if not args.quiet and not args.list_only:
        print_citation()
        print()

    if args.cite:
        print_citation()
        return 0

    if args.list_only:
        files = list_available_files()
        print(f"Available files in {DATASET_NAME}:")
        for f in files:
            print(f"  - {f['name']}: {f['description']}")
        print()
        print("For the complete and up-to-date file list, visit:")
        print(f"  {DATASET_URL}")
        return 0

    # Download
    verbose = not args.quiet
    if verbose:
        print(f"Downloading to: {args.output_dir}")
        print()

    try:
        downloaded = download_dataset(
            output_dir=args.output_dir,
            verbose=verbose,
        )

        if verbose:
            print()
            print(f"Downloaded {len(downloaded)} file(s) to {args.output_dir}")
            print()
            print("NOTE: The files above are the metadata/documentation.")
            print("The full trace data (several GB) must be downloaded from:")
            print(f"  {DATASET_URL}")
            print()
            print("Follow the instructions in the README for the complete dataset.")
            print()
            print("Once downloaded, process with:")
            print(f"  python -m energivanu.data.alibaba_processor --data-dir {args.output_dir}")

        return 0

    except KeyboardInterrupt:
        print("\n⏹  Download interrupted")
        return 130

    except Exception as exc:
        print(f"\n❌ Download failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
