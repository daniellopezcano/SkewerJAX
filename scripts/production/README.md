# Production runs

Many skewers through a simulation, computed once and stored, to be read back later: the analogue of notebook 11, kept on
disk. This directory is a self-contained tool **outside the package**: `skewerjax` holds the science and the engine,
and the code here composes its public API (`skewerjax.simulations`, `skewerjax.core`, `skewerjax.config`). Nothing
here changes what the engine computes.

| File | What it does |
|---|---|
| `run_production.py` | the launcher: one run, or a named sweep of runs, resumable, with a dry run (this README) |
| `production_runs.py` | one run: its configuration (`RunConfig`), its folder and record (`run.json`), its states, sharing, its execution |
| `production_sweeps.py` | the named sweeps (`points`, `pixels`, `skewers`, `simulations`, `smoke`) and `all` |
| `production_estimates.py` | the disk, time and memory of a run before it starts, from `benchmarks/results` |
| `skewer_store.py` | the skewer store: layout, identity (a hash of the options), nested lattices, writing, **reading** (`load_run`, `load_skewers`, `list_datasets`); NumPy and h5py only |
| `skewer_extraction.py` | the computation of a dataset's skewers: the options, the streaming route (`compute_skewers`), the sampled route (`compute_skewers_sampled`), resumable passes (`extract`) |
| `extract_skewers.py` | the store-level tool: several lattices and axes into one store (notebook 18's three lines of sight) |
| `STORE.md` | the format of the store, its identity rules, reading, checks |

Tests: `tests/production/` (fast; the end-to-end run on TNG50-4 is `slow` + `tng`). Notebooks 17 and 18 read stores
through `skewer_store.py` (`sys.path.insert(0, str(skewerjax.utils.paths.PRODUCTION_SCRIPTS_DIR))`).

## Quick start

```bash
cd /path/to/SkewerJAX                                              # the repository (any cwd works)
P=scripts/production/run_production.py
python $P --list                                                   # the sweeps and their runs
python $P --sweep smoke --out ~/skewerjax_production               # a 1-minute test of the launcher (TNG50-4)
python $P --sweep all --dry-run                                    # what the four checks would cost; writes nothing
nohup python $P --sweep all > ~/skewerjax_all.log 2>&1 &           # the four checks, in the background
python $P --status --sweep pixels                                  # the state of every run of a sweep
```

`--out` defaults to `$SKEWERJAX_PRODUCTION`, else `~/skewerjax_production`. It is never inside the repository (a
git-ignored folder there is accepted, e.g. a scratch folder of `claude_agents/`) nor inside a simulation's folder.

## One run

A run is one configuration (`production_runs.RunConfig`; the defaults are the benchmark setup):

| Field (`--flag`) | Default | Meaning |
|---|---|---|
| `sim` (`--sim`), `sim_path` | TNG50-2, `$SKEWERJAX_DATA/<sim>` | the simulation |
| `snap` | 25 | the snapshot |
| `lines` | HI-LyA SiIII-1206 | the lines; the first is the reference (the pixel axis) |
| `n` | 64 | skewers per transverse axis, a power of two: the n × n lattice x_i = i L / n, nested in every finer one |
| `axis` | 2 (y) | the line of sight: 1, 2, 3 = x, y, z |
| `dv_kms` or `n_pix` | 1 km/s | the pixel grid: target width (n_pix = int(span / dv)), or the number of pixels over one box |
| `points_kms` | none | the spacing of the sampling points: see "The points" below |
| `mode`, `options` (`--opt K=V`) | default | the preset (`default`: exact cell crossings, integrated profile, snapshot frame; `matched`: fake_spectra's scheme) and overrides of its options |
| `precision` | float32 | float32 or float64 (τ is stored as float32) |
| `device` | cpu | where the τ sum runs: cpu or gpu (the pass runs on the host) |
| `tile` | 1024 | skewers per tile, one HDF5 batch each |
| `threads` | every core | the pass's threads |

```bash
python $P --run lya_128 --sim TNG50-2 --n 128 --dv-kms 1          # -> <out>/single/lya_128/
python $P --run gpu_256 --n 256 --device gpu --tile 2048
python $P --run seg_0p25 --n 8 --points-kms 0.25 --opt pieces=segments --opt profile=midpoint
python $P --run x_axis --axis 1 --lines HI-LyA SiIII-1206 SiII-1193 SiII-1190
```

## Sweeps

A sweep is a named list of runs that vary **one** parameter around a baseline (`production_sweeps.py`), run one after
another, each in its own child process. Every run of a sweep is a folder `<out>/<sweep>/<run>/`, and the sweep has
`<out>/<sweep>/index.json` (every run: its configuration, state, estimate, dataset, timings) and `sweep.log`.

**The baseline** (`BASELINE`): TNG50-2 snapshot 25, Lyα + Si III 1206, 1 km/s pixels, along y, float32, CPU, 64² skewers
(the benchmark's 4,096-skewer point; 0.13 GB, ~1.7 min with compilation).

### The four resolution checks

| Sweep | Varies | Values | What it shows |
|---|---|---|---|
| `points` | the spacing of the sampling points | 0.125, 0.25, 0.5, 1, 2, 4 km/s (the benchmark's 0.25 among them), 8² skewers | how the sampled route converges to the integrated crossings (`segments_*` against `reference`), and that the a(s) table of the exact frame is converged at any spacing (`exact_*`) |
| `pixels` | the pixel width | 1, 2, 4, 8, 16 km/s | the effect of the output resolution on τ and its statistics (P1D up to k = π/dv, the mean flux, the line widths); compare the coarse grids to the 1 km/s run rebinned |
| `skewers` | the number of skewers | 4², 8², 16², 32², 64², 128², 256² (nested) | the convergence of the statistics (P1D, P(k, μ)) with the number of skewers (notebook 18), and, skewer by skewer, that a skewer's τ does not depend on the other skewers of its run (each lattice is a subset of every finer one) |
| `simulations` | the simulation | TNG50-2, TNG50-3, TNG50-4 (same lattice and pixels: the same skewer positions, one box size) | the convergence with the simulation's resolution, skewer by skewer at the same places |

`points` uses its own lattice, 8² (C2's, whose cost is measured): the sampled route selects the cells near every point
of every skewer (the gather route), which grows with the number of points; the spacing acts skewer by skewer, so 64
skewers of ~4,000 pixels measure it well. Every other field is the baseline's.

### The points

Under the default route (exact cell crossings, integrated profile, snapshot frame) **there are no sampling points**: each
cell crossing is one piece, its velocity integral exact, and the snapshot frame has one a for the whole box, so the
spacing controls nothing. It acts in two places, and the sweep varies it in both:

- **the exact frame** (`--opt frame=exact`): the a(s) table along the skewer (`stream.box_skewers`'s `frame_spacing`, default
  0.05 cMpc ≈ 3.8 km/s); a(s) is smooth, so the result should not depend on it (`exact_*` runs);
- **the sampled route** (`--opt pieces=segments --opt profile=midpoint`): the dual segments of points every `points_kms`
  (`cell_pieces(deposition="voronoi_segments")`, the midpoint profile: the scheme of the earlier default and of
  benchmark block C2), where the spacing sets the pieces and the accuracy (`segments_*` runs; C2 measured rms |ΔF| from
  9e-7 at 0.125 km/s to 3.5e-4 at 4 km/s for Lyα on 64 TNG50-2 skewers).

Both keys enter the dataset's options (and its hash) only where they act, so every other dataset keeps its hash.

### Cost (the dry run of 2026-11-21, this machine)

| Sweep | Runs | Disk | Wall time | Peak memory |
|---|---|---|---|---|
| `points` | 13 | 30 MB | 12 min | 5.8 GB |
| `pixels` | 5 | 0.26 GB | 8 min | 13.7 GB |
| `skewers` | 7 | 2.8 GB | 19 min (256²: 9.4 min, 2.1 GB) | 75 GB (256²; 37.7 GB up to 128²) |
| `simulations` | 3 | 0.39 GB | 3.4 min | 13.7 GB |
| `all` | 25 computed + 3 linked | 3.2 GB | 38 min | 75 GB |
| `smoke` | 2 | 0.8 MB | ~1 min | 5.7 GB |

The 256² run is split by the memory model into two passes (one pass would need 117 GB by the model; 65.7 GB were measured
in one pass): `--set memory_fraction=0.95` lets it run in one if the machine is free. **Recommended order**: `smoke`
first (the launcher on this machine), then `all` (skewers, simulations, pixels, points: the baseline 64² and the 8²
reference are computed once, in `skewers`, and linked into the other sweeps).

### `all`, sharing and resuming

`--sweep all` runs the four sweeps in `ALL_ORDER`. A run whose configuration (`RunConfig.key()`: every field that changes
what is computed, not the name, the threads or the memory fraction) is identical to a finished run anywhere under
`--out` is **linked** (hard links of its files: no extra disk) instead of recomputed (`--no-share` recomputes).

Resuming is rerunning the same command:

- a finished run (`run.json` status `done`, every file of its manifest present with its size) is skipped;
- an interrupted run (status `running` with its process gone, `launched`, `interrupted`), a failed one, or a damaged one
  (a file missing or of another size) is **redone from scratch** (its folder emptied first);
- `--force` recomputes finished runs too; without it, `--run NAME` on a finished run is refused (exit 3).

Interrupting: Ctrl-C, `kill <launcher pid>` (SIGTERM) or a closed terminal (SIGHUP) stop the launcher and its child; the
run's record says `interrupted` and the batches written stay until the run is redone.

```bash
nohup python $P --sweep skewers > ~/skewers.log 2>&1 &   # or: tmux new -s prod 'python scripts/production/run_production.py --sweep all'
tail -f ~/skewerjax_production/skewers/sweep.log          # the launcher's log; each run: <out>/<sweep>/<run>/run.log
kill <pid>                                                 # interrupt; rerun the same command to resume
python $P --sweep skewers --only n_0128 --force            # recompute one run
python $P --sweep pixels --set device=gpu --out ~/prod_gpu # every run of a sweep on the GPU (--set FIELD=VALUE)
```

## Output layout and reading back

```
<out>/
  <sweep>/                       points, pixels, skewers, simulations, smoke, or single (--run)
    index.json                   the sweep: every run, its configuration, state, estimate, dataset, timings
    sweep.log                    the launcher's log
    <run>/
      run.json                   the full configuration, status, package version and git commit, snapshot and cosmology,
                                 hardware, timings, the estimate, the manifest of the files (name: size)
      run.log                    the run's output
      store/<sim>/snap_<NNN>/axis_<a>/<hash>/
        config.json              the dataset's identity: location and every option of τ (STORE.md)
        skewers/batch_*.h5       indices (m, 2) int32, positions (m, 3) float64 [cMpc], tau/<line> (m, N_pix) float32
        runs/run_*.json          the extraction's record
```

Each run folder is self-describing: `run.json` and `config.json` are JSON, the batches HDF5, readable without this
directory. With it:

```python
import sys
from skewerjax.utils.paths import PRODUCTION_SCRIPTS_DIR
sys.path.insert(0, str(PRODUCTION_SCRIPTS_DIR))
from skewer_store import list_runs, load_run, load_skewers

st = load_run("~/skewerjax_production/skewers/n_0128")        # the run's lattice; load_run(dir, 32): a nested sub-lattice
st.tau, st.v, st.dv, st.L, st.u, st.positions, st.columns, st.config   # tau (n_lines, n^2, N_pix), a-major
[r["name"] for r in list_runs("~/skewerjax_production/pixels")]
# the store-level reader: load_skewers(run_dir / "store", sim, snap, axis, hash, n)  (hash: run.json["dataset"]["hash"])
```

**Notebook 17** loads one lattice of a store and checks it (the columns and positions are the lattice, τ finite and
≥ 0, the coarser lattice bit for bit the stride-2 subset): set `STORE` (or `$SKEWERJAX_STORE`) to `<run>/store` and its
`SIM`, `AXIS`, `N_PER_AXIS`. **Notebook 18** compares the statistics of nested lattices of one dataset (the store of
`extract_skewers.py` on three axes).

## Safety

- Before each run the free disk space is checked against its estimate × 1.1 plus `--reserve-gb` (default 10 GB): if it
  would not fit, the launcher refuses with a message (exit 3).
- One launcher per sweep folder (`.lock`); one writer per dataset (the store's lock).
- A run folder is emptied only if it holds a `run.json` (or nothing) inside `--out`.
- No persistent cache besides the outputs; the snapshots are only read.

## Estimates

`--dry-run` prints, per run, its disk size, wall time, peak memory and passes, and the totals; it writes nothing. How
they are derived (`production_estimates.py`):

- **disk**: n² (4 n_lines N_pix + 32) bytes (τ float32, indices, positions) + ~0.1 MB per batch, N_pix from the
  snapshot's header (exact);
- **time**: the measured totals of the scaling benchmark (`benchmarks/results/scaling/*.json`, cold page cache, CPU or
  GPU entries) at 4² to 256² TNG50-2 skewers, interpolated in log(skewers); the τ stage scaled to the pixel width by
  block C1; other simulations by their measured ratio to TNG50-2 at 128²; + the compilation of a fresh process (the
  record's `jit_s`) + one pass per extra pass; the sampled route: the pass + 2 × C2's τ time per skewer (rough);
  float64: τ doubled (not measured);
- **memory**: the pass model of `skewer_extraction` (5.7 GB + 92 B per cell within reach), the one that splits a run into
  passes; conservative against notebook 16 (37 / 29.8 GB modelled / measured at 128², 68 / 65.7 at 256²).

The smoke runs (TNG50-4, 2² to 32²) took 5–12 s against estimates of 22–42 s: the records are cold (page cache evicted)
and include the compilation of the benchmark's larger grids, so the estimates are upper bounds on a warm machine.
