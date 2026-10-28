# Tests

One pytest suite, laid out like the package and the pipeline (reader → cells → selection → pieces → τ → statistics):

| folder | covers |
|---|---|
| `core/` | constants and lines, cosmology a(x), skewers, cells, selection (in memory and streamed), deposition, pieces, spectral frames, Voigt, the optical depth (`test_optical_depth.py`: periodicity, frames, closed forms, precision, sharding, GPU), JAX transformations, statistics, input checks (`test_validation.py`) and diagnostics |
| `simulations/` | the GRF and its toy cells, gas physics, the TNG50-4 snapshot (reader, selections, field extraction, τ) |
| `utils/` | the figure functions |
| `comparison/` | the benchmark utilities, the comparison with fake_spectra |
| `regression/` | every result against reference arrays (`data/*.npz`, built by `outputs.py`) |

`conftest.py` runs JAX on the CPU in float64 and skips the tests whose markers need what the machine lacks;
`helpers.py` holds a few shared builders.

## Running it

```bash
pip install -e ".[dev,simulations]"        # pytest, pytest-cov, ruff, pre-commit, nbstripout, h5py

pytest                                     # the fast default: every test except the `slow` ones (pyproject.toml: -m "not slow")
pytest -m "slow or not slow"               # the full suite
pytest tests/core                          # one folder; pytest tests/core/test_optical_depth.py for one module
pytest -m tng                              # only the TNG50-4 tests (fast and slow)
pytest -m "not tng"                        # what the CI runs (no data, no GPU, no fake_spectra)
JAX_PLATFORMS=cuda,cpu pytest -m gpu       # the GPU tests
pytest -m fake_spectra                     # the comparison with fake_spectra (live; several minutes)
pytest --cov=skewerjax --cov-report=term   # with coverage
```

Markers (`pyproject.toml`):

| marker | meaning | without it |
|---|---|---|
| `slow` | minutes: full resolution scans, the whole-snapshot checks, fake_spectra live | deselected by default |
| `tng` | needs the TNG50-4 snapshots 25 and 33 in `$SKEWERJAX_DATA/TNG50-4` (default `/home/STORAGE`) | skipped, with the reason |
| `gpu` | needs a GPU visible to JAX (`JAX_PLATFORMS=cuda,cpu`) | skipped |
| `fake_spectra` | needs fake_spectra importable | skipped |

Without pytest (an environment where it cannot be installed), `python scripts/dev/run_tests.py [-m EXPR] [--all]
[path ...]` runs the same tests with a minimal stand-in (`scripts/dev/pytest_shim`), with the same markers and default.

## Regression references

`tests/regression/data/*.npz` hold the arrays of the code at the time they were written; every result must agree to
1e-10 of its scale (integers exactly). Regenerate them **only for an intended change of the results**, and record the
reason in `CHANGELOG.md`:

```bash
python scripts/dev/update_regression_references.py              # all of them; or: route_toy modules_toy modules_tng
```

It runs on the CPU and enables float64 before importing SkewerJAX, as the tests do (some import-time constants follow
that flag). `modules_tng` needs the TNG50-4 snapshot.

## Notebooks

The notebooks are not part of the suite. Run them headless from `notebooks/` with a fresh kernel
(`jupyter nbconvert --to notebook --execute <nb> --output-dir /tmp/out`), and check their markdown math with
`python scripts/dev/check_markdown_math.py notebooks/*.ipynb`.
