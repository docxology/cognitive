# tests

Pytest suite for the framework (`src/`) and the `Things/` prototypes.

## Layout

- `conftest.py` — session fixtures (`output_dir`, `temp_dir`, sample POMDP
  matrices); forces the headless `Agg` matplotlib backend.
- `test_matrix_ops.py` — `src.models.matrices.matrix_ops` (ops, loader,
  initializer).
- `test_visualization.py` — `src.visualization.matrix_plots` plotters.
- `test_simple_pomdp.py` — `Things.Simple_POMDP` model and expected free
  energy.
- `test_continuous_generic.py` — continuous-state dynamics
  (numpy/scipy/seaborn analysis only).
- `visualization/test_continuous_generic.py` — continuous active inference
  plus rendering; writes PNG/GIF artifacts to `Output/tests/visualization/`.
- `run_tests.py` — wrapper that runs pytest with coverage and a JUnit XML
  report, then writes `test_report.json` (ignored, not committed).
- `test_config.yaml` — configuration used by the POMDP tests.

## Running

From the repository root:

```bash
python -m pip install numpy pyyaml matplotlib seaborn scipy pytest pytest-cov tqdm click jinja2
python -m pytest tests -q      # ~6 min, 60 tests
python tests/run_tests.py      # coverage + JUnit XML wrapper
```

`pytest.ini` sets `testpaths = tests` and registers the `visualization`,
`basic`, `complex`, `multi_step`, and `single_step` markers.
