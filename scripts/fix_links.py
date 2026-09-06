#!/usr/bin/env python3
"""Thin orchestrator: analyze and repair [[wikilinks]] in knowledge_base/.

All analysis/fix/report logic lives in src/analysis/link_analysis.py
(LinkAnalyzer). Expects the inventory produced by list_file_directory.py
in the output directory (file_structure.json, files.csv, wikilinks.csv).
"""

import argparse
import sys
import time
from pathlib import Path

# Path bootstrap: make the repo root importable regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.link_analysis import LinkAnalyzer  # noqa: E402  (path setup must precede import)

from tqdm import tqdm  # noqa: E402


def parse_args(argv=None):
    """Parse CLI arguments (defaults reproduce the historical behavior)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).resolve().parent / 'output',
        help='Directory containing the inventory files (default: scripts/output)',
    )
    parser.add_argument(
        '--kb-root',
        type=Path,
        default=REPO_ROOT / 'knowledge_base',
        help='Knowledge-base root to fix (default: knowledge_base/)',
    )
    return parser.parse_args(argv)


def main():
    """Main function to analyze and fix links."""
    args = parse_args()
    output_dir = args.output_dir
    kb_root = args.kb_root

    start_time = time.time()

    print("\n🔍 Knowledge Base Link Analysis")
    print("=" * 40)

    print("\n⚡ Initializing link analyzer...")
    analyzer = LinkAnalyzer(output_dir)

    print("\n📊 Analyzing link issues...")
    with tqdm(total=3, desc="Progress", unit="step") as pbar:
        fixes = analyzer.suggest_fixes()
        pbar.update(1)

        print("\n🛠️  Applying fixes...")
        changes = analyzer.apply_fixes(fixes, kb_root)
        pbar.update(1)

        print("\n📝 Generating reports...")
        report = analyzer.generate_report(fixes, changes)
        summary = analyzer.generate_summary_report(fixes, changes)
        pbar.update(1)

    # Save reports
    report_file = output_dir / 'link_analysis_report.md'
    summary_file = output_dir / 'link_analysis_summary.md'
    report_file.write_text(report, encoding='utf-8')
    summary_file.write_text(summary, encoding='utf-8')

    end_time = time.time()
    duration = end_time - start_time

    print(f"\n✨ Analysis complete in {duration:.2f}s")
    print(f"📄 Full report saved to {report_file}")
    print(f"📑 Summary report saved to {summary_file}")
    print("\nSummary:")
    print(f"🔗 Fixed links: {sum(1 for c in changes if c['type'] == 'fixed_link')}")
    print(f"📁 Created files: {sum(1 for c in changes if c['type'] == 'created_file')}")
    print(f"🔄 Added backlinks: {sum(1 for c in changes if c['type'] == 'added_backlink')}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
