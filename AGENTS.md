# AGENTS.md — `cognitive`

Cognitive ecosystem modeling framework (Active Inference + Obsidian-style
knowledge base). Published as `docxology/cognitive`, default branch `main`.

## Layout

- `src/` — framework code: `models/active_inference/` (base model,
  dispatcher, homeostatic), `models/matrices/matrix_ops.py`,
  `analysis/` (`file_inventory.py`, `link_analysis.py`),
  `visualization/matrix_plots.py`, `utils/` (`matrix_utils.py`,
  `create_node.py`, `visualization/network_viz.py`).
- `Things/` — self-contained research prototypes, each with its own
  README/requirements where applicable: `Ant_Colony/`, `BioFirm/`,
  `Continuous_Generic/`, `Generic_POMDP/`, `Path_Network/`,
  `Simple_POMDP/`, `Baseball_Game/`. The full ant colony implementation is
  `Things/Ant_Colony/ant_colony/` (`ant_colony/main.py` is the CLI).
- `knowledge_base/` — concept nodes (cognitive, mathematics, biology,
  systems, technology, agents, BioFirm, evolution).
- `docs/` — Obsidian-oriented documentation; `docs/README.md` is the index.
- `templates/` — node templates used by `src/utils/create_node.py`.
- `tests/` — pytest suite (`tests/conftest.py`, `tests/run_tests.py`).
- `scripts/` — link tooling (`fix_links.py`, `list_file_directory.py`).
- `Output/` — committed test/visualization artifacts.
- `config.yaml` — project configuration (`test_mode` flag, active-inference
  parameters, knowledge-base paths).

## Setup and tests

No package manifest at the repo root; install the runtime/test deps and run
pytest from the repo root:

```bash
python -m pip install numpy pyyaml matplotlib seaborn scipy pytest pytest-cov tqdm click jinja2
python -m pytest tests -q
python tests/run_tests.py   # wrapper adding coverage + JUnit XML
```

Per-prototype extras live in `Things/*/requirements.txt`.

## Conventions and gotchas

- Most folder-level `AGENTS.md`/`README.md` files are auto-generated stubs
  marked `Status: SKELETON`. Their "local-only path under
  `projects/ongoing/` — never committed" scope note is **stale and wrong**:
  every one of those files is tracked here. Treat this root file as
  authoritative and expand a stub only when you work in that folder.
- Documentation uses wiki links (`[[path/to/node|Label]]`), so intra-doc
  links often do not resolve as filesystem paths; a bare name resolves to
  any unique file with that basename.
- `matplotlib` is imported with the `Agg` backend in `conftest.py` and
  several modules — keep tests headless.
- `Things/Path_Network` and `Things/Ant_Colony` require `torch`; the root
  `tests/` suite does not.
- `*.pyc`, `__pycache__/`, and `*.egg-info/` are ignored — never commit
  them.
