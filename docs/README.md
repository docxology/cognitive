# Cognitive Ecosystem Modeling Framework — Documentation

## What this repo is

A framework for modeling cognitive ecosystems using Active Inference,
integrated with Obsidian for knowledge management: agent behavior modeling,
knowledge-structure management, network visualization, and multi-agent
simulation (from `README.md` and `main.py`, which runs the ant colony
simulation via `ant_colony/`).

## Directory map

- `main.py` — entry point for the ant colony simulation
- `ant_colony/` — simulation core (`agents/`, `simulation.py`)
- `src/` — `models/`, `analysis/`, `visualization/`, `utils/`
- `config.yaml` — project configuration (active-inference parameters,
  knowledge-base settings, paths)
- `docs/` — documentation (Obsidian-oriented; see `docs/README.md`)
- `knowledge_base/`, `templates/` — knowledge-management assets
- `tests/` — simulation and POMDP/visualization tests (`run_tests.py`)

## How to run

Run commands as documented in the repo's own files (no package manifest
declares a CLI entry point, so commands below are from file contents):

```bash
python main.py                      # ant colony simulation (argparse-driven)
python tests/run_tests.py           # test runner script in tests/
```

Configuration (inference iterations, learning rate, knowledge-base linking)
is set in `config.yaml`.

## How to use this docs/ directory

`docs/README.md` indexes the subfolders (concepts, guides, api, examples,
tools, research, templates). This file is the human entry point;
`docs/AGENTS.md` holds agent-facing conventions.

## Status

Not documented in repo — needs owner input (no release/version or
publication milestone is stated in tracked files).
