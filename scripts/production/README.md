# Production runs

Many skewers through a simulation, computed once and stored, to be read back later: the analogue of notebook 11, kept on
disk. This directory is a self-contained tool **outside the package**: `skewerjax` holds the science and the engine,
and the code here composes its public API (`skewerjax.simulations`, `skewerjax.core`, `skewerjax.config`). Nothing
here changes what the engine computes.

| File | What it does |
|---|---|
| `run_production.py` | the launcher: one run, or a named sweep of runs, one at a time, resumable, with a dry run and a memory check (this README) |
| `production_runs.py` | one run: its configuration (`RunConfig`), its folder and record (`run.json`), its states, its execution in a child process |
| `production_sweeps.py` | the resolution campaign: `FIDUCIAL`, the sweeps `fiducial`, `points`, `pixels`, `skewers`, `simulations`, `smoke`, and `all` |
| `production_estimates.py` | the disk, time and memory of a run before it starts, from `benchmarks/results` |
| `skewer_store.py` | the skewer store: layout, identity (a hash of the options), nested lattices, writing, **reading** (`load_run`, `load_skewers`, `list_datasets`); NumPy and h5py only |
| `skewer_extraction.py` | the computation of a dataset's skewers: the options, the streaming route (`compute_skewers`), the sampled route (`compute_skewers_sampled`), resumable passes sized by a memory model (`extract`) |
| `extract_skewers.py` | the store-level tool: several lattices and axes into one store |
| `STORE.md` | the format of the store, its identity rules, reading, checks |

Tests: `tests/production/` (fast; the end-to-end runs on TNG50-4 are `slow` + `tng`). Notebooks 17–21 read the store
through `skewer_store.py` (`sys.path.insert(0, str(skewerjax.utils.paths.PRODUCTION_SCRIPTS_DIR))`).

## Quick start

```bash
cd /path/to/SkewerJAX                                              # the repository (any cwd works)
P=scripts/production/run_production.py
python $P --list                                                   # the sweeps and their runs
python $P --sweep smoke                                            # a 1-minute test of the launcher (TNG50-4)
python $P --sweep all --dry-run                                    # what the campaign would cost; writes nothing
setsid nohup python $P --sweep all > ~/skewerjax_production/campaign.log 2>&1 < /dev/null &   # detached, survives the session
python $P --status --sweep all                                     # the state of every run
tail -f ~/skewerjax_production/campaign.log                        # the launcher; each run: <out>/<sweep>/<run>/run.log
```

`--out` defaults to `$SKEWERJAX_PRODUCTION`, else `~/skewerjax_production`. It is never inside the repository (a
git-ignored folder there is accepted, e.g. a scratch folder of `claude_agents/`) nor inside a simulation's folder.

## One run

A run is one configuration (`production_runs.RunConfig`; the defaults are the benchmark setup) and one lattice:

| Field (`--flag`) | Default | Meaning |
|---|---|---|
| `sim` (`--sim`), `sim_path` | TNG50-2, `$SKEWERJAX_DATA/<sim>` | the simulation |
| `snap` | 25 | the snapshot |
| `lines` | HI-LyA SiIII-1206 | the lines; the first is the reference (the pixel axis) |
| `n` | 64 | skewers per transverse axis, a power of two: the n × n lattice x_i = i L / n, nested in every finer one |
| `axis` | 2 (y) | the line of sight: 1, 2, 3 = x, y, z |
| `dv_kms` or `n_pix` | 1 km/s | the pixel grid: nominal width (n_pix = int(span / dv), dv = span / n_pix: the grid tiles the velocity period exactly), or the number of pixels over one box |
| `points_cMpc` or `points_kms` | none | the spacing of the sampling points along a skewer, in cMpc or in km/s of Hubble flow: see the `points` sweep |
| `mode`, `options` (`--opt K=V`) | default | the preset (`default`: exact cell crossings, integrated profile, snapshot frame; `matched`: fake_spectra's scheme) and overrides of its options (e.g. `frame=exact`, `pieces=segments`, `profile=midpoint`) |
| `precision` | float32 | float32 or float64 (τ is stored as float32) |
| `device` | cpu | where the τ sum runs: cpu or gpu (the pass runs on the host) |
| `tile` | 1024 | skewers per tile, one HDF5 batch each |
| `threads` | every core | the pass's threads |
| `memory_fraction` | 0.6 | of the available memory one pass may use (the memory model sizes the passes) |

```bash
python $P --run lya_128 --sim TNG50-2 --n 128 --dv-kms 1 --opt frame=exact --points-cMpc 0.05   # -> <out>/single/lya_128/
python $P --run gpu_256 --n 256 --device gpu --tile 2048
python $P --run seg_0p01 --n 32 --points-cMpc 0.01 --opt pieces=segments --opt frame=exact
python $P --run x_axis --axis 1 --lines HI-LyA SiIII-1206 SiII-1193 SiII-1190
```

## The resolution campaign

Four independent checks, each varying **one** parameter around a common **fiducial run** (`production_sweeps.py`), to
learn what sampling, pixels, number of skewers and simulation resolution P1D and the P(k, μ) wedges need to be
converged to 0.5 % (notebooks 18–21).

**The fiducial run** (`FIDUCIAL`, sweep `fiducial`): TNG50-2 snapshot 25, Lyα + Si III 1206, the 128 × 128 lattice along
y, the sampling spacing 0.05 cMpc, 1 km/s pixels, float32, CPU, the default route (the exact cell crossings, the
integrated profile, the exact frame). With the crossings the spacing is only the a(s) table of the exact frame; it is set
explicitly and recorded. The fiducial is computed once: every run that coincides with it reads its dataset.

| Sweep | Simulation | Sampling dx [cMpc] | Pixel width [km/s] | Skewers per side | Notebook |
|---|---|---|---|---|---|
| `points` | TNG50-2 | **1, 0.5, 0.1, 0.05, 0.01** | 1 | 128 | 18 |
| `pixels` | TNG50-2 | 0.05 | **10, 5, 1, 0.5, 0.1** | 128 | 19 |
| `skewers` | TNG50-2 | 0.05 | 1 | **32, 64, 128, 256, 512** | 20 |
| `simulations` | **TNG50-2, TNG50-3, TNG50-4** | 0.05 | 1 | 128 | 21 |

- **`points`**: the sampling points define the pieces. The five runs use `pieces="segments"`: the cells crossed by the
  skewer, cut at the midpoints between consecutive points (the dual segments; `cell_pieces(deposition=
  "voronoi_segments")`, the gather route `compute_skewers_sampled`), with the **integrated profile** (the package default;
  `--opt profile=midpoint` is the option of benchmark block C2) and the exact frame, its a(s) table on the same points.
  This is what a user with any skewer, curved ones included, gets. The fiducial run (`points/reference`: the exact
  crossings, no sampling in the pieces) is the converged reference.
- **`pixels`**: nominal widths; the actual width (the period over an integer number of pixels) is recorded per run.
- **`skewers`**: the five lattices are one dataset, the fiducial's: `n_0256` and `n_0512` **extend** it (only the
  skewers it does not hold are computed), `n_0032`, `n_0064` and `n_0128` are its sub-lattices, bit for bit.
- **`simulations`**: TNG50-2, -3 and -4 have one box size (35 cMpc/h) and snapshot 25 is z = 3.008 in the three, so the
  lattice puts the skewers at the same comoving positions and the pixels on the same velocities (checked in notebook 21).

`--sweep all` runs `ALL_ORDER`, cheap and foundational first: the fiducial; `simulations` (TNG50-3, TNG50-4); `pixels`
(10, 5, 0.5, 0.1 km/s); `points` (1, 0.5, 0.1, 0.05, 0.01 cMpc: coarse to fine); `skewers` (256², then 512², the
heaviest). A run that fails is recorded as failed and the next one starts; the failures are listed at the end.

`smoke`: TNG50-4, 2² then 4² (the second extends the first's dataset) and one sampled 4² run, 8 km/s: the launcher in a
minute.

### Cost (the dry run of 2026-11-21, this machine: 125 GB available, 534 GB free)

| Sweep | Runs computed (reused) | Skewers computed | Disk | Wall time (estimate) | Peak (model) |
|---|---|---|---|---|---|
| `fiducial` | 1 | 16,384 | 0.52 GB | 3.9 min | 37.7 GB |
| `simulations` | 2 (1) | 32,768 | 1.04 GB | 4.8 min | 20.2 GB |
| `pixels` | 4 (1) | 65,536 | 6.38 GB | 25 min | 42.4 GB |
| `points` | 5 (1) | 81,920 | 2.60 GB | 3.4 h (rough) | 75 GB |
| `skewers` | 2 (3) | 245,760 | 7.80 GB | 55 min | 75 GB |
| `all` | 14 (6) | 442,368 | 18.3 GB | 4.9 h | 75 GB |

The sampled route's estimate is rough (calibrated on one tile); the launcher recalibrates every estimate by the runs
that finish (the median of measured / estimated, per route).

### Actual (the campaign of 2026-11-21/22)

| Run | Estimate | Actual wall | Peak RSS (`ru_maxrss`) |
|---|---|---|---|
| `fiducial` | 3.9 min | 2.5 min | 30.2 GB |
| `simulations/TNG50-3`, `TNG50-4` | 2.5, 2.1 min | 50 s, 26 s | 16.8, 6.3 GB |
| `pixels/dv_10kms`, `dv_5kms`, `dv_0p5kms`, `dv_0p1kms` | 3.4, 3.5, 4.9, 12.7 min | 2.1, 2.1, 2.6, 4.9 min | 31.1, 31.0, 30.2, 31.5 GB |
| `points/dx_1cMpc` | 40.5 min | ≈ 4.7 h over 5 launches (see below) | — |
| `points/dx_0p5cMpc` | 40.9 min | 102.8 min | 126.5 GB (file-backed maps included) |
| `points/dx_0p1cMpc`, `dx_0p05cMpc`, `dx_0p01cMpc` | 31.6, 30.3, 60.5 min | 30.3, 32.4, 61.4 min | ~127 GB (idem) |
| `skewers/n_0256`, `n_0512` | 8.3, 45.6 min | 7.0, 18.9 min | 33.8, 35.0 GB |
| reuse runs (6) | 15 s | 1 s each | 0.7 GB |

The streaming route ran at 0.5–0.8 × its estimate. The sampled route's cost grows with the reach of the dual segments
(2.5 h + dx / 2): at dx = 1 and 0.5 cMpc a pass selects 60–165 M cells and every skewer has ~5 M candidate generators
(about 0.65 s per skewer in tiles of 1,024; tiles of ~1,850 cost 1.2 s per skewer), 3–7 × the estimate; at dx ≤ 0.1 cMpc
the estimate held. `ru_maxrss` counts the memory-mapped snapshot (file-backed, reclaimable): the anonymous memory of a
sampled pass was 17–30 GB at dx ≤ 0.1 cMpc and ~70 GB for 1,800 skewers at 1 cMpc. `dx_1cMpc` was restarted twice to
change its tile and pass sizes (stored batches kept), failed once on an engine error that did not recur on retry
(`core/deposition.py`, `_voronoi_envelope`: arrays of 5,100,298 and 5,100,297 candidates, one element apart, that come
from one mask), and completed on the retry.

## Runs share datasets; resuming

Every run is an n × n lattice of a dataset of the campaign's store `<out>/store` (the dataset: the hash of the resolved
options, `STORE.md`). Runs whose options resolve to the same dataset share it:

- a run identical to a finished one, or a coarser lattice of a finer one, **computes nothing** (action `reuse`);
- a finer lattice **extends** the dataset (action `extend`): only the skewers it does not hold are computed.

Resuming is rerunning the same command:

- a finished run (`run.json` status `done`, every batch file of its lattice present with its size) is skipped;
- an interrupted run (status `running` with its process gone, `launched`, `interrupted`), a failed one, or a damaged one
  (a batch file missing or of another size) is **redone**: the extraction resumes from the batches already in the store
  (each batch is written atomically: only the tiles in progress are lost);
- `--force` deletes the run's dataset (every run on it loses its skewers; the log names them) and recomputes it; without
  it, `--run NAME` on a finished run is refused (exit 3).

Interrupting: Ctrl-C, `kill <launcher pid>` (SIGTERM) or a closed terminal (SIGHUP) stop the launcher and its child; the
run's record says `interrupted`. After a crash or a power cut, the same command finishes the campaign.

```bash
python $P --sweep all --status                             # every run: state, wall time, skewers computed, folder
tail -f ~/skewerjax_production/points/sweep.log             # one sweep; each run: <out>/<sweep>/<run>/run.log
kill <pid>                                                  # interrupt; rerun the same command to resume
python $P --sweep pixels --only dv_0p5kms --force           # recompute one run (its dataset is deleted first)
python $P --sweep pixels --set device=gpu --out ~/prod_gpu  # every run of a sweep on the GPU (--set FIELD=VALUE)
```

## Output layout and reading back

```
<out>/
  store/<sim>/snap_<NNN>/axis_<a>/<hash>/   the campaign's skewer store, one dataset per configuration (STORE.md)
    config.json                  the dataset's identity: location and every option of τ
    skewers/batch_*.h5           indices (m, 2) int32, positions (m, 3) float64 [cMpc], tau/<line> (m, N_pix) float32
    runs/run_*.json              the record of every extraction into it
  <sweep>/                       fiducial, points, pixels, skewers, simulations, smoke, or single (--run)
    index.json                   the sweep: every run, its configuration, state, estimate, dataset, sampling, timings
    sweep.log                    the launcher's log
    <run>/
      run.json                   the full configuration, status; the dataset and lattice it reads ("dataset": root
                                 ../../store, hash, n_per_axis); the sampling and pixels as computed ("sampling": dx in cMpc
                                 and km/s, what it controls, the route, the nominal and actual dv); the package version and
                                 git commit, the snapshot and cosmology, the hardware, the timings (skewers computed and
                                 reused), the peak memory, the estimate, the manifest of the batch files of its lattice
      run.log                    the run's output
```

Everything is JSON and HDF5, readable without this directory. With it:

```python
import sys
from skewerjax.utils.paths import PRODUCTION_SCRIPTS_DIR
sys.path.insert(0, str(PRODUCTION_SCRIPTS_DIR))
from skewer_store import list_runs, load_run, load_skewers

st = load_run("~/skewerjax_production/skewers/n_0512")        # the run's lattice; load_run(dir, 32): a nested sub-lattice
st.tau, st.v, st.dv, st.L, st.u, st.positions, st.columns, st.config   # tau (n_lines, n^2, N_pix), a-major
[r["name"] for r in list_runs("~/skewerjax_production/pixels")]
# the store-level reader: load_skewers("~/skewerjax_production/store", sim, snap, axis, hash, n)
```

**Notebook 17** loads one lattice of the store and checks it (default: the fiducial run's dataset, 128²). **Notebooks
18–21** compare the runs of each sweep with its reference (`summary_stats.run_statistics`, `compare_runs`,
`k_max_table`: the mean transmission, P1D and the P(k, μ) wedges, the relative difference per bin with jackknife errors,
the largest k converged to 0.1, 0.5 and 1 %).

## Safety

- **One production process at a time**: the launcher computes one run after another, each in its own child process.
- **Memory**: before each run the available memory (MemAvailable) is checked against the run's smallest pass (one tile)
  × 1.2 + 4 GB; while it does not fit the run waits (logged every 10 minutes); then the child sizes its passes to
  `memory_fraction` of the memory available (more passes when less is free). A pass never holds more skewers than the
  model allows (`skewer_extraction.bounded_tiles`): the squares of neighbouring skewers are cut when the skewers left by
  an interrupted run are uneven. For a sampled run at dx ≥ 0.5 cMpc, `--set memory_fraction=0.12` keeps its passes small
  (the model is a bound measured at 1 cMpc, below).
- **Disk**: before each run the free space is checked against its estimate × 1.1 plus `--reserve-gb` (default 10 GB): if
  it would not fit, the launcher refuses with a message (exit 3).
- One launcher per sweep folder (`.lock`); one writer per dataset (the store's lock).
- A run folder is cleared only if it holds a `run.json` (or nothing) inside `--out`; a dataset is deleted (`--force`) only
  inside `<out>/store`.
- No persistent cache besides the outputs; the snapshots are only read.

## Estimates

`--dry-run` prints, per run, its action, the skewers it computes, its disk size, wall time, peak memory and passes, and
the totals with the free disk and RAM; it writes nothing. How they are derived (`production_estimates.py`):

- **disk**: the new skewers × (4 n_lines N_pix + 32) bytes (τ float32, indices, positions) + ~0.1 MB per batch, N_pix
  from the snapshot's header (exact);
- **time**, the streaming route: the measured totals of the scaling benchmark (`benchmarks/results/scaling/*.json`, cold
  page cache) at 4² to 256² TNG50-2 skewers, interpolated in log(skewers); the τ stage scaled to the pixel width by
  block C1 (below 1 km/s as 1 / dv: rough); other simulations by their measured ratio to TNG50-2 at 128²; + the
  compilation + one pass per extra pass;
- **time**, the sampled route: per pass the gather selection (85 s on TNG50-2 + 1 µs per selected cell), then 27 µs per
  piece (a skewer's points + its ~1,200 cell crossings), measured on one 1,024-skewer tile at 0.01 cMpc: rough;
- **calibration**: every estimate is multiplied by the median measured / estimated of the finished runs of its route that
  computed at least 1,024 skewers;
- **memory**: the streaming pass model (5.7 GB + 92 B per cell within reach), conservative against notebook 16 (37 / 29.8
  GB modelled / measured at 128², 68 / 65.7 at 256²); the sampled route: 4 GB + 500 B per selected cell (every cell within
  2.5 h + dx / 2 of a point, bounded by the patch of neighbouring skewers of a pass) + 50 B per point, the per-cell bytes
  measured at 1 cMpc in the campaign (~73 GB for 143 M cells; 250 B held at 0.01 cMpc).
