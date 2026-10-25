# Cognitive Ecosystem Modeling Framework — Documentation

## What this repo is

A framework for modeling cognitive ecosystems using Active Inference,
integrated with Obsidian for knowledge management: agent behavior modeling,
knowledge-structure management, network visualization, and multi-agent
simulation. Framework code lives in `src/`; runnable prototypes (including
the ant colony simulation) live under `Things/`.

## Directory map

- `src/` — `models/`, `analysis/`, `visualization/`, `utils/`
- `Things/` — self-contained prototypes; the ant colony simulation is
  `Things/Ant_Colony/ant_colony/` (CLI: `ant_colony/main.py`)
- `config.yaml` — project configuration (active-inference parameters,
  knowledge-base settings, paths)
- `docs/` — documentation (Obsidian-oriented; this file is the index)
- `knowledge_base/`, `templates/` — knowledge-management assets
- `tests/` — simulation and POMDP/visualization tests (`run_tests.py`)
- `scripts/` — link and inventory tooling

## How to run

No package manifest declares an installable distribution, so install the
libraries the code and tests import:

```bash
python -m pip install numpy pyyaml matplotlib seaborn scipy pytest pytest-cov tqdm click jinja2
```

```bash
python -m pytest tests -q        # test suite from the repo root
python tests/run_tests.py        # wrapper: coverage + JUnit XML report
cd Things/Ant_Colony && python -m ant_colony.main   # reference simulation (adds torch, networkx, h5py, pandas)
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
