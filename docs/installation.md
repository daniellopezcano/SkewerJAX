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
pip install -e ".[simulations]"     # + h5py: the snapshot readers
pip install -e ".[plots]"           # + tol-colors, ipympl (optional figure extras)
pip install -e ".[notebooks]"       # + Jupyter, ipympl, ipywidgets, nbconvert for the notebooks
pip install -e ".[comparison]"      # + h5py, fake-spectra (PyPI), psutil: the benchmark; for notebook 12 see below
pip install -e ".[dev]"             # + pytest, pytest-cov, ruff, pre-commit, nbstripout, nbconvert, psutil
pip install -e ".[gpu]"             # + jax[cuda12]
```

Extras can be combined, e.g. `pip install -e ".[simulations,notebooks,dev]"`. `import skewerjax` needs only the core
dependencies; the snapshot readers raise an `ImportError` for h5py without the `simulations` extra. After a
development install, run `pre-commit install` once (ruff, nbstripout and the file checks on every commit).

### Optional dependencies

- **fake_spectra** is needed only by notebook 12, the tests with the `fake_spectra` marker
  (`tests/comparison/test_fake_spectra.py`, `tests/core/test_fake_spectra_options.py`) and the benchmark; the package never
  imports it elsewhere. Install it as in the next section, in one environment with SkewerJAX.
- **tol_colors** (Paul Tol's colour schemes) is optional: `skewerjax.utils.plotting` takes its "bright" scheme from it
  when it is installed and otherwise uses a vendored copy with the same values.
- **ipympl** gives interactive figures in a live notebook (`plotting.use_backend()`); without it, figures are inline.
- **CLOUDY ionisation tables** for metal ions:
  - the tables are expected in `skewerjax/simulations/cloudy_tables/ion_out_photo_atten`;
  - `cloudy_tables` is a git submodule (sbird/cloudy_tables, the same tables as fake_spectra's): clone with
    `--recurse-submodules` or run `git submodule update --init`; the package data of `pyproject.toml` includes them.

## fake_spectra in one environment (notebook 12)

Notebook 12 and the benchmark run fake_spectra natively next to SkewerJAX. The recipe is `envs/skewerjax-fs.yml`
(Python 3.11, JAX 0.6.2 with the CUDA 12 plugin, the validated versions of the scientific stack), then, from the
repository root:

```bash
conda env create -f envs/skewerjax-fs.yml
conda activate skewerjax-fs
git -C .project_material/fake_spectra submodule update --init --recursive   # its CLOUDY tables, 121 MB
pip install .project_material/fake_spectra                                  # non-editable: builds with gcc and OpenMP
pip install -e ".[simulations,notebooks,dev]"                               # not ".[comparison]": that takes fake-spectra from PyPI
python -m ipykernel install --user --name skewerjax-fs --display-name skewerjax-fs
```

`.project_material/fake_spectra` is any clone of [sbird/fake_spectra](https://github.com/sbird/fake_spectra) (validated:
`ff5844d`, version 2.3.0). The first `pip install` fails with `package directory 'fake_spectra/cloudy_tables/ion_out_no_atten'
does not exist` until the submodule is initialised; the `project.license` message before it is only a deprecation
warning. The build needs a C++ compiler with OpenMP: it found the system `gcc` here (`setup.py` falls back to one thread
without OpenMP, with a warning). HDF5 comes with h5py: its C reader (`cextract`) is not built.

Check it:

```bash
python -c "import skewerjax, fake_spectra"
python -m pytest "$(python -c 'import fake_spectra, os; print(os.path.dirname(fake_spectra.__file__))')/tests" -o addopts=""   # 33 tests
JAX_PLATFORMS=cpu python -m pytest -m "slow or not slow"                    # from the repository root
```

## 4. Simulation data

SkewerJAX reads Arepo/TNG HDF5 snapshots with `ArepoHDF5Reader(base, snapnum)`, where `base` is the directory that
contains `snapdir_XXX/`. Paths are always passed explicitly in code. The notebooks, tests and benchmarks take every path
outside the package from one module, `skewerjax.utils.paths`, each overridable by an
environment variable:

```bash
export SKEWERJAX_DATA=/path/to/folder/with/TNG50-4   # DATA_DIR: holds TNG50-4/ (and TNG50-3/, TNG50-2/ for the benchmark)
export SKEWERJAX_BENCHMARK_RAW=/path/to/raw          # BENCHMARK_RAW_DIR: per-configuration benchmark records
export SKEWERJAX_BENCHMARK_RESULTS=/path/to/records  # BENCHMARK_RECORDS_DIR: the summary notebook 15 reads
```

Notebooks 08 and 10 recompute their toy boxes every time it runs and stores nothing.

`DATA_DIR` defaults to `/home/STORAGE` (read only). The benchmark folders default to `benchmarks/raw` (one record per
configuration in `benchmarks/raw/fake_spectra`, git-ignored) and `benchmarks/results`
(`fake_spectra_benchmark.json`, the summary of every record that notebook 15 reads, committed); both are written by
`scripts/benchmarks/benchmark_fake_spectra.py`.

## 5. Verify the installation

```bash
python scripts/checks/check_install.py                            # dependencies, JAX devices, the package imports
python scripts/checks/check_toy.py                                # a toy box end to end (a few seconds)
python scripts/checks/check_tng.py                                # one TNG50-4 skewer end to end (skipped without the data)
pytest                                                            # the fast test suite (needs the dev extra)
pytest -m "slow or not slow"                                      # every test, incl. the slow physics checks
```

The tests that need the TNG50-4 snapshots (`tng`), a GPU (`gpu`) or fake_spectra (`fake_spectra`, environment `skewerjax-fs`) skip themselves
when those are missing (`tests/README.md`).

## Troubleshooting

- **Precision.** float32 by default; `skewerjax.config.set_precision("float64")` (or `SKEWERJAX_PRECISION=float64`) for
  float64, which turns `jax_enable_x64` on (`docs/precision.md`).
- **All the cores.** `skewerjax.config.configure_devices(n_cpu_devices=..., platforms="cuda,cpu")` before the first
  `import jax`; then `devices=skewerjax.config.devices("cpu")` or `("gpu")` in `compute_optical_depth`.
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
