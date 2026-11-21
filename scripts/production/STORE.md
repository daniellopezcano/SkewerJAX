# The skewer store

A store of skewers on disk for convergence tests and long production runs: the optical depth τ of every line along
nested N × N lattices of skewers through a snapshot, computed by `scripts/production/extract_skewers.py` (one store,
from a terminal, resumable) or by `run_production.py` (runs and sweeps, one store per run folder: `README.md`), and read
by `load_skewers` (notebook 17; the statistics and their convergence: notebook 18). Doubling the number of skewers per
axis reuses every skewer already computed; another simulation, snapshot, axis, pixel width or option is another dataset
of the same tree. The code, in this directory (production code, outside the package): `skewer_store.py` (layout,
identity, reading; NumPy and h5py only, so a store is readable without SkewerJAX) and `skewer_extraction.py` (options,
computation, runs).

## 1. Layout

```
<out>/                                   the root, --out (never inside the repository nor the simulation's folder)
└── <sim>/snap_<NNN>/axis_<a>/<hash>/    one dataset: a simulation, a snapshot, a line of sight, a set of options
    ├── config.json                      its identity, human readable
    ├── skewers/batch_<run>_<k>.h5       the skewers, in batches (one HDF5 file per tile of the computation)
    ├── runs/run_<run>.json              one record per invocation of the script
    └── .lock                            held while a run writes (one writer per dataset)
```

**A batch** holds `indices` (m, 2) int32 (the skewers on the finest lattice, below), `positions` (m, 3) float64 [cMpc]
(the point of each skewer at line-of-sight coordinate 0), `tau/<line_id>` (m, N_pix) float32, and attributes: the
hash, the run, the git commit (and whether the tree had changes), the precision, the device and the resolution check
of the batch. Only τ is stored: F = e^−τ, δ and every statistic follow from it; the densities along the skewers are not
kept (notebook 11's density figures are not reproduced from the store).

**Writing is atomic.** A batch is written to a temporary file (`.tmp_batch_…`), flushed to disk (`fsync`) and renamed:
a crash leaves either the whole batch or a temporary file. The index of the stored skewers is rebuilt by scanning the
batches (`SkewerDataset.index`), so no shared file is ever rewritten; temporary files are ignored, and deleted by the
next writer (who holds the lock); a batch that cannot be read (truncated by hand, another hash) is reported and its
skewers count as missing.

**Disk.** 4 bytes per pixel and line: 31.7 kB per skewer for two lines on 1 km/s pixels of TNG50 (3,961 pixels);
128² skewers 0.52 GB per axis, 256² 2.1 GB, 512² 8.3 GB, 1024² 33 GB. `--dry-run` prints the size of a run and the free
space, and the script stops before a run that would not fit.

## 2. The identity of a dataset

`config.json` has four parts:

| part | content | in the hash |
|---|---|---|
| `simulation` | name (`--sim`), path, snapshot, axis | no: the folder names it (checked when a dataset is opened) |
| `options` | every input that changes τ (below), resolved to values | **yes** |
| `snapshot` | redshift, a, h, Ω_m, L, u = aH, the number of gas cells and of files | no: follows from the location |
| `preset`, `hash`, `created`, `layout_version` | the preset the options came from (`default`, `matched`), … | no |

`options`, all explicit (no `None` standing for "the package's default", no `"snapshot"` standing for a number, so
the hash keeps its meaning whatever the defaults of the package become):

- `lattice`: `n_ref` (the finest lattice, below), the offset 0, the order of the skewers;
- `pixels`: `spacing_kms` (the requested width) and the resulting `n_pix` = int(span / spacing), `dv_kms` = span / n_pix,
  `period_kms` = n_pix dv (the span of one box along the skewer, (aH)_snap L in the snapshot frame), the centres
  (i + 1/2) dv; `--n-pix` gives the number of pixels instead;
- `lines`: `line_ids` (the first the reference: every line on its axis with `line_frame="reference"`);
- `physics`: the options of `gas_to_cells` (mass unit, constants of T, x_HI model, CLOUDY density and interpolation);
- `pieces`: `crossings` (the exact Voronoi cell crossings) or `chords` (fake_spectra's top-hat chords, `chord_weight`);
- `tau`: frame (`snapshot` or `exact`), line frame, profile integration, Voigt, detuning, Hubble flow, (aH)_ref, a_eff,
  b constants, pixel average, tautail, wing tolerance and the period (`velocity_period`);
- `precision`: of the computation (τ is stored as float32 in both).

The hash is the first 10 hex digits of the SHA-256 of the canonical JSON of `options`. The location is not in it: the
three lines of sight of one configuration share their hash (notebook 18 loads them by one hash), and the same options
on TNG50-4 and TNG50-2 differ only through `n_pix` when the pixel width is the same. A dataset whose `simulation.path`
changed is accepted with a warning when its header (cells, box, redshift) is the same (the data moved), refused
otherwise (another simulation under the same name: use another `--sim`). The device, the number of threads, the batch
size and the passes do not enter the identity: they do not change τ beyond rounding (the GPU against the CPU, ΔF ≤ 1e-5
in float32; a run interrupted and resumed is bit for bit the uninterrupted one) and are recorded in `runs/`.

**The presets** (`skewer_extraction.MODES`, `--mode`): `default` is notebook 11's τ (the snapshot frame, linear Hubble
flow and detuning, periodic over one box, the lines on the Lyα axis) with the package's pieces and integration (the
exact crossings, the integrated profile, Harris' Voigt, CODATA constants, SkewerJAX's gas physics); `matched` is
fake_spectra's scheme (`comparison.scaling.mode_options`: its gas physics, top-hat chords of weight 1/8, the midpoint
profile, `pixel_average`, `tautail` 1e-7, a_eff and (aH)_ref of the snapshot, its constants) on the same frame and axes.
Notebook 11 uses the same Voigt (Harris, the default of `compute_optical_depth`); `--set voigt=ExoJAX` gives ExoJAX's.

## 3. The nested lattice

On the two transverse axes (a, b) of a line of sight (x: (y, z), y: (x, z), z: (x, y)) the skewers sit at
x_i = i L / N, i = 0 … N − 1, with N a power of two. A skewer is named by its integer indices (i_a, i_b) on the finest
lattice N_ref (`--n-ref`, a power of two, default 2^12 = 4,096, 12.6 kpc apart on TNG50's 51.7 cMpc box), so the
N × N lattice is the subset with stride N_ref / N (`skewer_store.lattice_indices`), ordered a-major as
`core.lattice_columns(N, (1, 1))` and `summary_stats.on_lattice`.

**Why powers of two.** Doubling N must keep every skewer of N: the nodes i L / N are nodes of the 2N lattice
(2i L / 2N). Any N that divides N_ref nests in the finer lattices that are its multiples, but a series such as 3, 4, 6
shares almost nothing (3 and 4 have only i = 0 in common), and an index on one grid for every lattice needs every N to
divide N_ref. Powers of two satisfy both: every lattice of a series is a subset of the next, and of N_ref. Another N is
refused (`--n-per-axis 3`: an error before anything runs).

## 4. Running

```bash
python scripts/production/extract_skewers.py --help
```

| option | meaning |
|---|---|
| `--sim`, `--sim-path`, `--snap`, `--axis` | the simulation (TNG50-2, -3, -4 under `$SKEWERJAX_DATA`, or any name with `--sim-path`), the snapshot, one or several lines of sight (run in turn) |
| `--n-per-axis` | the lattices (powers of two), run in increasing order |
| `--spacing-kms` or `--n-pix` | the pixel grid |
| `--lines` | the line ids (default Lyα and Si III 1206) |
| `--mode`, `--set KEY=VALUE` | the preset and any option of it (`--set voigt=ExoJAX`, `--set physics.x_HI_model=stored`, `--set tautail=1e-7`) |
| `--precision`, `--n-ref` | float32 (default) or float64; the finest lattice |
| `--device`, `--batch`, `--threads` | where the τ sum runs (`cpu`: two JAX devices on every core; `gpu`); skewers per batch (a tile of the streaming route, default 1,024); threads of the pass |
| `--skewers-per-pass`, `--memory-fraction` | the passes over the snapshot (default: as many skewers as fit 0.6 of the available memory) |
| `--out`, `--dry-run`, `--config` | the root; print the plan only; a YAML or JSON file of these options (the command line overrides it) |

**What a run does.** For each N, increasing: the skewers of the N × N lattice not stored yet (the coarser lattices of
the series are by then) are computed by the streaming route of `simulations.stream` (notebook 15, `docs/performance.md`):
one pass over the snapshot finds the cells within reach of those skewers and runs the gas physics on them, then the
skewers are summed tile by tile, and each tile is written as one batch as soon as its τ is known. When the skewers of
one N need more memory than the budget, they are split into several passes, neighbouring skewers together. The model
of a pass (notebook 16): 5.7 GB + 92 B per cell within reach, at most min(n_gas, n × 2.5²π n_gas^(1/3)) cells for n
skewers (29.7 GB at 128² TNG50-2 skewers; 29.8 GB measured). On this machine (125 GiB) the default budget, 76 GB, takes
128² TNG50-2 skewers in one pass and the 49,152 new skewers of 256² in two. The log gives every pass and every batch
with the progress over the whole run, the rate and the ETA; `runs/run_<id>.json` keeps it.

**Time** (TNG50-2, 1 km/s, Lyα + Si III, cold, `docs/performance.md`): a pass costs 27 s at 16 skewers, 55 s at 1,024,
81 s at 128², 161 s at 256²; τ adds 3.7 ms per skewer on the CPU, 1.9 ms on the GPU (6.3 ms in the matched mode). A
doubling series makes one pass per N (each on the new skewers only): 2 … 128 is 7 passes and 16,384 skewers, about
6 minutes per axis on the CPU by these numbers (an estimate, not a measured run).

**Examples.**

```bash
# a doubling series on TNG50-2, the three lines of sight (in tmux: survives a closed terminal)
tmux new -s skewers
python scripts/production/extract_skewers.py --sim TNG50-2 --snap 25 --axis 1 2 3 \
    --n-per-axis 2 4 8 16 32 64 128 --spacing-kms 1.0 --out ~/skewer_store 2>&1 | tee -a ~/skewer_store/tng50-2.log
# (Ctrl-b d detaches; tmux attach -t skewers comes back)

# the next doubling later: only the 49,152 new skewers of each axis are computed
python scripts/production/extract_skewers.py --sim TNG50-2 --axis 1 2 3 --n-per-axis 256 --spacing-kms 1.0 --out ~/skewer_store

# another pixel width (another dataset, another hash), on the GPU
python scripts/production/extract_skewers.py --sim TNG50-2 --axis 2 --n-per-axis 2 4 8 16 32 64 --spacing-kms 4.0 \
    --device gpu --out ~/skewer_store

# another simulation, fake_spectra's scheme, a dry run first
python scripts/production/extract_skewers.py --sim TNG50-3 --axis 2 --n-per-axis 64 128 --mode matched --out ~/skewer_store --dry-run
```

A configuration file (`--config run.yaml`) holds the same options:

```yaml
sim: TNG50-2
snap: 25
axis: [1, 2, 3]
n_per_axis: [2, 4, 8, 16, 32, 64, 128]
spacing_kms: 1.0
lines: [HI-LyA, SiIII-1206]
mode: default
set: {voigt: ExoJAX}
out: /data/skewer_store
```

**Interrupting and resuming.** Ctrl-C, a kill (SIGTERM) or a closed terminal (SIGHUP) stops the run: the batches
written stay, the tile in progress is lost, and the run's record says `interrupted`. The same command resumes: it reads
what is stored and computes the rest (checked: a run of 16² TNG50-4 skewers killed after 4 of 16 batches and resumed is
bit for bit the uninterrupted run). Two runs on the same dataset at once are refused (the lock); runs on different
datasets (another axis, another pixel width) can run side by side if the memory allows (each makes its own pass).

**`--dry-run`** prints the snapshot, the dataset (its path, whether it exists, its full `config.json`), for each N the
skewers, those stored and those to compute, the disk needed and free, and the passes with their estimated peak memory.

## 5. Reading

```python
import sys
from skewerjax.utils.paths import PRODUCTION_SCRIPTS_DIR   # scripts/production: the reader is not part of the package
sys.path.insert(0, str(PRODUCTION_SCRIPTS_DIR))           # (or the path of this directory, without SkewerJAX)
from skewer_store import format_datasets, list_datasets, load_run, load_skewers, select_datasets

rows = list_datasets("~/skewer_store")          # one row per dataset: sim, snap, axis, hash, preset, n_pix, n_stored, n_max
print(format_datasets(rows))
sel = select_datasets("~/skewer_store", "TNG50-2", 25)   # one configuration on its axes: hash, axes, n_values (complete on every axis)
st = load_skewers("~/skewer_store", "TNG50-2", 25, 2, sel["hash"], 128)
st.tau            # (n_lines, 128^2, N_pix) float32, the skewers a-major (st.columns: (i_a, i_b) on the 128 lattice)
st.v, st.dv, st.period, st.u, st.L, st.line_ids, st.positions, st.config
F = np.exp(-st.tau)
st = load_run("~/skewerjax_production/pixels/dv_2kms")   # a run folder of run_production.py (its run.json names the dataset)
```

`load_skewers(root, sim, snap, axis, config_or_hash, n_per_axis)` takes the hash, a unique prefix of it, or a config;
it raises a `ValueError` naming the largest complete N when the requested lattice is not complete.
`SkewerDataset.open(...)` gives the lower level: `index()`, `missing(n)`, `complete(n)`, `largest_complete()`,
`runs()`. The statistics of notebook 11 from a stored lattice are one call, `summary_stats.flux_statistics(tau, dv,
lattice_grid(...), line_ids=...)`, and their convergence `summary_stats.lattice_convergence` (notebook 18; `log=` reports each step of `flux_statistics`). `SkewerDataset.sources(n)` lists the levels of the nested lattices an n x n lattice is read from (notebook 17).

## 6. Checks

- `tests/production/test_skewer_store.py` (float32 and float64; temporary directories only): the lattices nest; N = 2
  then 4 computes the 12 new skewers only, and the N = 2 lattice loaded afterwards is bit for bit the earlier one; an
  interrupted run and a batch interrupted before its rename leave no corrupt file and are redone; a corrupt batch is
  ignored and recomputed; one writer at a time; another option, pixel width or N_ref is another dataset, another axis
  the same hash; τ and the metadata round-trip; the presets are the comparison modes; on TNG50-4 (`tng`), N = 2 then 4 by
  the streaming route against notebook 11's pipeline (max |ΔF| < 5e-5 in float32, 2e-6 in float64, along y) and against a direct
  N = 4 run.
- τ against notebook 11's pipeline (every cell read, `gas_to_cells`, `cell_pieces` on `SimulationSkewers.axis_aligned`,
  `compute_optical_depth`, the same options) on the 16 TNG50-4 skewers of N = 4 along each axis: in float32 max |ΔF|
  2.2e-5, 2.0e-5, 8.5e-5 (x, y, z), max |Δτ| 4.4e-6 of the largest τ of the skewers (τ_max = 2,220, 881, 5,020): the
  float32 floor set by the strongest absorber; in float64 max |ΔF| 4.3e-7 (y), 2.7e-7 (z) (the streaming route keeps
  n, T and v in float32, and τ is stored as float32).

## 7. Extending

- **A new option of τ or of the gas physics**: add it to the presets (`skewer_extraction.MODES`) and to `dataset_options`
  (resolved to a value) and pass it in `compute_skewers`: it enters `options`, hence the hash, automatically, and
  `--set` reaches it at once. The datasets computed before keep their hash and stay readable (their `options` lack the
  key), but a run of the new code names a new dataset (its options have the key) and starts it from nothing: the old
  skewers are not reused under the new option, as they should not be if the option changes τ.
- **Other lines**: `--lines` (any ids of `core.lines`; the first is the reference); another set of lines is another
  dataset.
- **Another simulation**: any snapshot `ArepoHDF5Reader` reads, with `--sim <name> --sim-path <folder>`.
- **Another stored quantity** (e.g. the densities along the skewers): a new dataset in the batches and a new
  `LAYOUT_VERSION`; readers check the hash and the shapes of what they read.
- **Other paths** (not axis-aligned, other offsets): the identity of a skewer is its index on the lattice; another
  geometry needs another index (and the gather route of `simulations.snapshot_optical_depth`).
