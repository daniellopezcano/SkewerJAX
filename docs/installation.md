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

From the repository root:

```bash
pip install -e .                    # core dependencies (JAX for the CPU included)
pip install -e ".[gpu]"             # the same, plus jax[cuda12]
pip install -e ".[notebooks]"       # Jupyter, ipympl and ipywidgets for the notebooks
pip install -e ".[comparison]"      # fake-spectra (PyPI) and psutil for the comparison with fake_spectra and the benchmark
pip install -e ".[dev]"             # pytest, nbformat/nbconvert (headless notebook runs), psutil
```

Extras can be combined, e.g. `pip install -e ".[gpu,notebooks,comparison,dev]"`.

### Optional dependencies

- **fake_spectra** is needed only by `skewerjax.comparison`: its live mode, `scripts/validation/check_fake_spectra.py`
  and the benchmark. It is published on PyPI as `fake-spectra` (2.3.1 at the time of writing; the development
  environment uses an editable 2.2.4 clone).
  - Without it, the comparison subpackage still works from a reference HDF5 file (`FakeSpectraSource(reference=...)`).
  - The rest of the package never imports it.
- **jztree / jaxkd** (JAX neighbour search) are *not* used. The cell selection uses SciPy's `cKDTree`, which was the
  faster option on the CPU (see the reply of session 20261021_2330).
- **CLOUDY ionisation tables** for metal ions:
  - the tables are expected in `skewerjax/simulations/cloudy_tables/ion_out_photo_atten`;
  - `cloudy_tables` is a git submodule (`git submodule update --init`);
  - the large table files are not tracked by git.

## 4. Simulation data

SkewerJAX reads Arepo/TNG HDF5 snapshots with `ArepoHDF5Reader(base, snapnum)`, where `base` is the directory that
contains `snapdir_XXX/`. Paths are always passed explicitly in code. The validation scripts, tests and the
benchmark read their snapshots from one environment variable:

```bash
export SKEWERJAX_DATA=/path/to/folder/with/TNG50-4   # holds TNG50-4/ (and TNG50-3/, TNG50-2/ for the benchmark)
```

It defaults to `/home/STORAGE`. The notebooks 04, 06, 08 and 09 read `SKEWERJAX_DATA` with the same default.

## 5. Verify the installation

```bash
python scripts/verify_install.py                                  # environment, devices, a tiny optical depth
python scripts/validation/run_all.py --quick --skip tng,fake      # the toy-input validation scripts (~2 min)
pytest                                                            # tests/ (needs the dev extra)
```

With TNG50-4 available, `python scripts/validation/run_all.py` runs every check, including the live comparison with
fake_spectra.

## Troubleshooting

- **Precision.** SkewerJAX follows JAX's precision flag and never sets it itself. The validated tolerances are float64
  ones, so call `jax.config.update("jax_enable_x64", True)` before computing. Without it JAX computes in float32. The
  τ kernel can also be asked for float32 on its own with `dtype="float32"` (see `docs/usage.md`).
- **Forcing the CPU.** Set `JAX_PLATFORMS=cpu` before Python starts (or before `import jax`). The validation scripts and
  tests do this themselves.
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
  not add compute. `tests/test_precision_and_sharding.py` does this in a subprocess.
