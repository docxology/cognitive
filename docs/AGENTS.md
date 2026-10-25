# Cognitive — Agent Notes

## Layout

- Framework code: `src/` (models, analysis, visualization, utils).
- Simulation logic: the ant colony package lives in
  `Things/Ant_Colony/ant_colony/` (agents incl. `Nestmate`,
  `simulation.py`, CLI `main.py`); other prototypes are sibling
  directories under `Things/`.
- Configuration: `config.yaml` (paths, active_inference parameters,
  knowledge_base settings; note `test_mode` flag).
- Documentation style is Obsidian-first: `docs/` uses `[[wiki links]]`
  (e.g. `[[active_inference|Active Inference]]`), so intra-doc links may
  not resolve as filesystem paths — navigate via `docs/README.md`.
- Tests live in `tests/` with `conftest.py` and `run_tests.py`.

## Conventions observed

- Wiki-link documentation (`[[...]]`) rather than Markdown-relative links.
- No package manifest (no pyproject.toml/setup.py); Python + numpy +
  PyYAML imports are declared at module level.

## How docs here are maintained

`docs/README.md` is the index; add new topic files under the matching
subfolder (concepts/guides/api/examples/tools/research/templates) and link
them from the index using the repo's wiki-link style. `DOCUMENTATION_ROADMAP.md`
in `docs/` tracks planned documentation work.
