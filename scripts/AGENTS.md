# AGENTS.md — `cognitive/scripts`

Technical specification for this project's thin orchestrator scripts.

## Script Inventory

| Script | Purpose | Delegates to | Command |
|--------|---------|--------------|---------|
| `list_file_directory.py` | File/mimetype/wikilink inventory with JSON/CSV/TXT export | `src/analysis/file_inventory.py` | `python scripts/list_file_directory.py` |
| `fix_links.py` | Detect and repair broken/ambiguous/missing-backlink wikilinks | `src/analysis/link_analysis.py` (`LinkAnalyzer`) | `python scripts/fix_links.py` |

Non-executable: `link_fix_prompt.md` (LLM prompt template).

## Design Contract

- Scripts are orchestration only: argparse, path bootstrap (`Path(__file__)`
  resolves repo root onto `sys.path`), progress printing, and a single
  delegated call into `src/<pkg>/` entrypoints.
- Business/data/analysis logic lives in `src/` and is importable + testable;
  scripts must not implement it inline.
- Default CLI arguments reproduce the historical in-repo behavior
  (inventory target = repo root, artifacts under `scripts/output/`).
- `fix_links.py` mutates `knowledge_base/` when run; run the inventory script
  first and review the generated report before applying fixes to the real
  knowledge base.

## Gotchas

- `fix_links.py` requires the three inventory files (`file_structure.json`,
  `files.csv`, `wikilinks.csv`) produced by `list_file_directory.py` in the
  same `--output-dir`; it fails loudly if they are missing.
- The repo has no `pyproject.toml`; run scripts/tests from the repository root
  so that `src.*` imports resolve.
