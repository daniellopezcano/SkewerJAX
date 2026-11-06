# Tests

One pytest suite, laid out like the package and the pipeline (reader → cells → selection → pieces → τ → statistics):

| folder | covers |
|---|---|
| `core/` | constants and lines, cosmology a(x) and r_d, skewers and the lattice of skewers (`test_skewer_lattice.py`), cells, selection (in memory and streamed), deposition, pieces, spectral frames, Voigt, the optical depth (`test_optical_depth.py`: periodicity, frames, a line shift beyond the period, closed forms, precision, sharding, GPU), JAX transformations (of the whole package), input checks (`test_validation.py`) and diagnostics |
| `summary_stats/` | the overflux and its statistics: P1D / ξ1D / P3D recovery, the decomposition by order and pair, contamination fits, the 3D grid statistics (`test_statistics.py`); P and ξ of gas cells through the mesh against P(k) (`test_field_stats.py`); the integral constraint, the BAO maxima and the Fourier imprint of a pair of lines (`test_theory.py`); the density on a lattice of skewers against its exact expectation (`test_density.py`); the one path of the field statistics, the cells of every source through a mesh (the same calls for toy and Voronoi cells, convergence to the exact grid FFT at low k, mass conservation, redshift-space cells), the statistics of skewer samples refusing `mas` and `interlace`, never calling the mesh (a spy, with a positive control), their line-of-sight sizes (`test_point_statistics.py`); the decomposition of N = 3 and 4 lines, the periodic ranges (`test_statistics.py`) |
| `simulations/` | the GRF and its toy cells, the toy boxes (and, through the explicit pipeline of notebook 10, along skewers: `test_toy_boxes.py`), gas physics, the section of the Voronoi cells by a plane (`test_maps.py`), the TNG50-4 snapshot (reader, selections, field extraction, τ) |
| `utils/` | the figure functions (any number of lines; the BAO images in a periodic box); the architecture specification (`test_dag.py`: every module, function and import it names exists, the layers are the import graph, the README's generated blocks are current) |
| `comparison/` | the benchmark utilities, the comparison with fake_spectra |
| `regression/` | every result against reference arrays (`data/*.npz`, built by `outputs.py`) |
| `test_mesh.py` | mass assignment and interlacing (`summary_stats.mesh`): mass conservation and the periodic wrap, the nodes bit for bit, the window, plane waves below and above k_Nyq, the sign of the interlacing phase, jit and gradients, in 1D, 2D, 3D; the chunked deposit and `node_transfer` |
| `test_import_graph.py` | the import graph: `simulations` and `summary_stats` import `core` at most and not each other (static, on the source, and at run time) |

`conftest.py` runs JAX on the CPU in float64 and skips the tests whose markers need what the machine lacks;
`helpers.py` holds a few shared builders (test-only), among them `toy_tau` and `toy_lattice_statistics`: the toy boxes
along a lattice of skewers by the explicit pipeline of notebook 10.

## Running it

```bash
pip install -e ".[dev,simulations]"        # pytest, pytest-cov, ruff, pre-commit, nbstripout, h5py

pytest                                     # the fast default: every test except the `slow` ones (pyproject.toml: -m "not slow")
pytest -m "slow or not slow"               # the full suite
pytest tests/core                          # one folder; pytest tests/core/test_optical_depth.py for one module
pytest -m tng                              # only the TNG50-4 tests (fast and slow)
pytest -m "not tng"                        # what the CI runs (no data, no GPU, no fake_spectra)
JAX_PLATFORMS=cuda,cpu pytest -m gpu       # the GPU tests
pytest -m fake_spectra                     # fake_spectra natively (environment skewerjax-fs; several minutes)
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
