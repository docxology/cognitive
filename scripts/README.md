# scripts

Thin orchestrator scripts for the cognitive ecosystem framework. Testable logic
lives in `src/analysis/`; scripts only set paths, parse arguments, and invoke
module entrypoints.

## Scripts

### `list_file_directory.py`

Scans a directory tree and inventories files (name, extension, size, MIME
type, mtime) plus the Obsidian `[[wikilink]]` graph found in `.md`/`.txt`
content. Exports the inventory in JSON, CSV, and text formats.

```bash
python scripts/list_file_directory.py [--target DIR] [--output-dir DIR]
```

Defaults: `--target` = repository root, `--output-dir` = `scripts/output/`.

Module logic:

- `src/analysis/file_inventory.py` — wikilink extraction, file metadata
  collection, and JSON/CSV/text export

Outputs (`scripts/output/`):

- `file_structure.json` — full inventory with wikilink graph and stats
- `files.csv`, `wikilinks.csv` — tabular file list and source→target link edges
- `file_structure.txt` — human-readable summary

### `fix_links.py`

Consumes the inventory exported by `list_file_directory.py`, finds broken,
ambiguous, and backlink-less wikilinks in `knowledge_base/`, applies fixes
(relink to best suggestion, create template files, add backlinks), and writes
Markdown reports.

```bash
python scripts/fix_links.py [--output-dir DIR] [--kb-root DIR]
```

Defaults: `--output-dir` = `scripts/output/` (must contain the inventory
files), `--kb-root` = `knowledge_base/`.

Module logic:

- `src/analysis/link_analysis.py` — `LinkAnalyzer`: issue detection, fix
  suggestion/application, and report generation

Outputs (`scripts/output/`):

- `link_analysis_report.md` — detailed issue list and applied changes
- `link_analysis_summary.md` — per-issue-type summary with samples

### `link_fix_prompt.md`

Prompt template for LLM-assisted manual link fixing; not executable.
