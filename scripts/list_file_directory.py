#!/usr/bin/env python3
"""Thin orchestrator: inventory repository files, mimetypes, and [[wikilinks]].

All scanning/export logic lives in src/analysis/file_inventory.py.
"""

import argparse
import sys
import time
from pathlib import Path

# Path bootstrap: make the repo root importable regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.file_inventory import (  # noqa: E402  (path setup must precede import)
    export_structure,
    list_files_in_directory,
    print_structure,
)


def parse_args(argv=None):
    """Parse CLI arguments (defaults reproduce the historical behavior)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--target',
        type=Path,
        default=REPO_ROOT,
        help='Directory to inventory (default: repository root)',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).resolve().parent / 'output',
        help='Directory for exported reports (default: scripts/output)',
    )
    return parser.parse_args(argv)


def main():
    """Main function to run the directory analysis."""
    args = parse_args()
    start_time = time.time()

    print("\n🔍 Knowledge Base Directory Analysis")
    print("=" * 40)

    print(f"\n📂 Analyzing directory: {args.target}")
    structure = list_files_in_directory(args.target)

    export_structure(structure, args.output_dir)

    # Also print to console
    print_structure(structure)

    end_time = time.time()
    duration = end_time - start_time
    print(f"\n✨ Analysis completed in {duration:.2f}s\n")


if __name__ == "__main__":
    main()
