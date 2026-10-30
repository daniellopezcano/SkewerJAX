# Installation

SkewerJAX is a pure-Python package on top of JAX. It runs on the CPU by default and on NVIDIA GPUs when a CUDA build of
JAX is installed.

## Supported versions

| component | version | source |
|---|---|---|
| Python | 3.11 or newer (validated with 3.11.14) | `requires-python` in `pyproject.toml`; SciPy 1.16 requires 3.11 |
| JAX / jaxlib | 0.6.2 or newer (validated with 0.6.2) | `pyproject.toml` |
| NumPy | 1.26.4 or newer, below 2 | ExoJAX 2.2.2 declares `numpy<2` in its package metadata |
| OS | Linux (validated on Debian, kernel 6.1) | the CUDA wheels of JAX are Linux x86_64 / aarch64 only |

The remaining lower bounds in `pyproject.toml` are the versions the package was validated with: SciPy 1.16.3,
h5py 3.15.1, matplotlib 3.10.8, diffrax 0.7.0 and ExoJAX 2.2.2.

## 1. Create an environment

With `venv`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

With conda (the development environment is a conda environment named `skewerjax`):

```bash
conda create -n skewerjax python=3.11 pip
conda activate skewerjax
```

Install JAX and the scientific stack with pip only, inside either environment. Mixing a conda-forge JAX with the pip
CUDA plugin is not supported.

## 2. Install JAX

The commands come from the official JAX installation guide, https://docs.jax.dev/en/latest/installation.html. The names
of the CUDA 12 extras were checked against the metadata of the installed `jax` 0.6.2 (`cuda12`, `cuda12-local`).

CPU only:

```bash
pip install --upgrade "jax>=0.6.2"
```

NVIDIA GPU, with the CUDA libraries from pip wheels (needs an NVIDIA driver >= 525 on Linux for CUDA 12):

```bash
pip install --upgrade "jax[cuda12]>=0.6.2"
```

NVIDIA GPU, with a locally installed CUDA 12 toolkit and cuDNN:

```bash
pip install --upgrade "jax[cuda12-local]>=0.6.2"
```

The current guide recommends CUDA 13 (`jax[cuda13]`) for new JAX releases. SkewerJAX has only been validated with
JAX 0.6.2 and CUDA 12.

## 3. Install SkewerJAX (editable)

Clone with the submodule of the CLOUDY tables (`git clone --recurse-submodules ...`, or `git submodule update --init`
in an existing clone), then, from the repository root:

```bash
pip install -e .                    # the core (JAX for the CPU included): toy fields, tau, statistics, figures
pip install -e ".[simulations]"     # + h5py: the snapshot readers and the result cache
pip install -e ".[plots]"           # + tol-colors, ipympl (optional figure extras)
pip install -e ".[notebooks]"       # + Jupyter, ipympl, ipywidgets, nbconvert for the notebooks
pip install -e ".[comparison]"      # + h5py, fake-spectra (PyPI), psutil: the comparison with fake_spectra and the benchmark
pip install -e ".[dev]"             # + pytest, pytest-cov, ruff, pre-commit, nbstripout, nbconvert, psutil
pip install -e ".[gpu]"             # + jax[cuda12]
```

Extras can be combined, e.g. `pip install -e ".[simulations,notebooks,dev]"`. `import skewerjax` needs only the core
dependencies; the snapshot readers raise an `ImportError` for h5py without the `simulations` extra. After a
development install, run `pre-commit install` once (ruff, nbstripout and the file checks on every commit).

### Optional dependencies

- **fake_spectra** is needed only by `skewerjax.comparison`: its live mode, the tests with the `fake_spectra` marker
  (`tests/comparison/test_fake_spectra.py`) and the benchmark. It is published on PyPI as `fake-spectra` (2.3.1 at the time of writing; the development
  environment uses an editable 2.2.4 clone).
  - Without it, the comparison subpackage still works from a reference HDF5 file (`FakeSpectraSource(reference=...)`).
  - The rest of the package never imports it.
- **jztree** (JZ-TREE, CUDA-only neighbour search with periodic boxes) is optional: the GPU back end of the cell
  selection (`select_cells_near_paths(..., backend="jztree" | "auto")`). It is not installed by default, and CPU-only
  installations never import it; the default is SciPy's `cKDTree`. Its CUDA 12 wheels (`pip install "jztree[cuda12]"`,
  jztree 1.1.0) require jax ≥ 0.8 and NumPy ≥ 2.2.6, which conflict with this environment (jax 0.6.2, NumPy < 2 for
  ExoJAX): check with `pip install --dry-run` first, or use a separate environment.
- **tol_colors** (Paul Tol's colour schemes) is optional: `skewerjax.utils.plotting` takes its "bright" scheme from it
  when it is installed and otherwise uses a vendored copy with the same values.
- **ipympl** gives interactive figures in a live notebook (`plotting.use_backend()`); without it, figures are inline.
- **CLOUDY ionisation tables** for metal ions:
  - the tables are expected in `skewerjax/simulations/cloudy_tables/ion_out_photo_atten`;
  - `cloudy_tables` is a git submodule (sbird/cloudy_tables, the same tables as fake_spectra's): clone with
    `--recurse-submodules` or run `git submodule update --init`; the package data of `pyproject.toml` includes them.

## 4. Simulation data

SkewerJAX reads Arepo/TNG HDF5 snapshots with `ArepoHDF5Reader(base, snapnum)`, where `base` is the directory that
contains `snapdir_XXX/`. Paths are always passed explicitly in code. The notebooks, tests and benchmarks take every path
outside the package from one module, `skewerjax.utils.paths`, each overridable by an
environment variable:

```bash
export SKEWERJAX_DATA=/path/to/folder/with/TNG50-4   # DATA_DIR: holds TNG50-4/ (and TNG50-3/, TNG50-2/ for the benchmark)
export SKEWERJAX_BENCHMARK_RAW=/path/to/raw          # BENCHMARK_RAW_DIR: per-configuration benchmark records
export SKEWERJAX_BENCHMARK_RESULTS=/path/to/records  # BENCHMARK_RECORDS_DIR: the summary notebook 10 reads
```

Notebook 08 recomputes its toy boxes every time it runs and stores nothing.

`DATA_DIR` defaults to `/home/STORAGE` (read only). The benchmark folders default to `benchmarks/raw` (one record per
configuration in `benchmarks/raw/fake_spectra`, git-ignored) and `benchmarks/results`
(`fake_spectra_benchmark.json`, the summary of every record that notebook 10 reads, committed); both are written by
`scripts/benchmarks/benchmark_fake_spectra.py`.

## 5. Verify the installation

```bash
python scripts/verify_install.py                                  # environment, devices, a tiny optical depth
pytest                                                            # the fast test suite (needs the dev extra)
pytest -m "slow or not slow"                                      # every test, incl. the slow physics checks
```

The tests that need the TNG50-4 snapshots (`tng`), a GPU (`gpu`) or fake_spectra (`fake_spectra`) skip themselves
when those are missing (`tests/README.md`).

## Troubleshooting

- **Precision.** SkewerJAX follows JAX's precision flag and never sets it itself. The validated tolerances are float64
  ones, so call `jax.config.update("jax_enable_x64", True)` before computing. Without it JAX computes in float32. The
  τ kernel can also be asked for float32 on its own with `dtype="float32"` (see `docs/usage.md`).
- **Forcing the CPU.** Set `JAX_PLATFORMS=cpu` before Python starts (or before `import jax`). The tests do this
  themselves (`JAX_PLATFORMS=cuda,cpu pytest -m gpu` for the GPU tests).
- **GPU memory.** By default JAX preallocates 75 % of the GPU memory on first use. Set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` to allocate on demand (the benchmark does), or
  `XLA_PYTHON_CLIENT_MEM_FRACTION=.50` to change the fraction
  (https://docs.jax.dev/en/latest/gpu_memory_allocation.html).
- **"Unable to load cuSPARSE" / "a CUDA-enabled jaxlib is not installed", although `nvidia-smi` sees the GPU.** The
  pip CUDA wheels are installed, but the dynamic loader does not find their libraries. Prepend them to
  `LD_LIBRARY_PATH` before starting Python:

  ```bash
  export LD_LIBRARY_PATH="$(python -c 'import site, glob; print(":".join(sorted(glob.glob(site.getsitepackages()[0] + "/nvidia/*/lib"))))'):${LD_LIBRARY_PATH:-}"
  python -c "import jax; print(jax.devices())"      # [CudaDevice(id=0)]
  ```

  On a conda environment, the same line can go into `$CONDA_PREFIX/etc/conda/activate.d/` so that it runs on every
  activation.
- **Several CPU devices for testing the sharding.** Set
  `XLA_FLAGS=--xla_force_host_platform_device_count=4` before JAX starts. It splits the host into 4 devices; it does
  not add compute. `tests/core/test_optical_depth.py` does this in a subprocess.
