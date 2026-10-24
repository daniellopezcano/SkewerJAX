# Usage guide

This guide is the reference companion of the notebooks. It lists the entry points of each subpackage, the conventions they
share and the options that control speed and memory. The notebooks (`notebooks/00_quickstart`, `00_quickstart_step_by_step`, `01`–`08`) show
the same tools at work, step by step.

```mermaid
flowchart LR
    subgraph A["core"]
        L[LineBundle, Cosmology] --> S[SkewerBundle]
        S --> LS["pieces_from_fields → Pieces"]
        F[Fields] --> LS
        LS --> T["compute_optical_depth → optical_depth_from_pieces / _sharded"]
        T --> O[DeltaFieldBuilder] --> ST[OverfluxStatistics1D / 3D]
    end
    subgraph B["simulations"]
        R[ArepoHDF5Reader] --> SS[SimulationSkewers] --> AB["Absorbers → absorber_pieces"] --> TA[tau_absorbers]
    end
    subgraph C["comparison"]
        FS[FakeSpectraSource] --- MT[matched_tau] --- BM[benchmark]
    end
    TA --> T
    MT --> SS
```

Dependencies run one way only: `comparison` uses `simulations`, `simulations` uses `core`, and nothing in `core`
imports the other subpackages. `skewerjax.utils.plotting` is outside them.

## Conventions

- **Units.** Comoving Mpc for positions and path lengths, km/s for velocities, K for temperatures. Number densities are
  comoving cm⁻³ in `core.Fields` and physical cm⁻³ at the data epoch in `simulations.Absorbers`. Wavelengths are in Å.
- **Shapes.** τ is `(N_lines, N_skewers, N_v)`; a `SkewerBundle` holds `(N_cosmologies, N_skewers, N_points, 4)` events
  (a, x, y, z).
- **Pieces.** τ is a sum over pieces, fragments of the path within which the gas is constant, each evaluated at its
  centroid (a, a⁻², the line centre and the fields). Simulation outputs give the exact intersections of the path with
  the cells; fields known everywhere give the dual segment of every sampling point (midpoint to midpoint), or
  `subdivide=q` equal pieces of it. Both reach the kernel as `core.Pieces` through `compute_optical_depth`.
- **Periodicity** is one option of `compute_optical_depth`, `velocity_period`: `False` (default, the direct sum),
  `True` (each skewer periodic over its own velocity span, set by a at its first and last points) or a period in km/s.
  The detuning is then replaced by its nearest image; only that image is kept. `SimulationSkewers.tau`,
  `optical_depth` and `tau_absorbers` pass it through; nothing else wraps a spectrum.
- **Frames.**
  - Each line is integrated in its own line-centred velocity frame v_† (`line_frame="own"`, default), or directly on
    the reference line's axis (`line_frame="reference"`: line j at v + Δv_j).
  - `wrap_to_ref_line` shifts spectra computed in their own frames onto the reference axis by interpolation: without a
    period, τ is NaN where the shift leaves the grid; with `velocity_period=P` (a spectrum computed periodic on a grid of
    one period), it wraps modulo P.
  - In simulations, `frame="exact"` uses a(x) along the path and `frame="snapshot"` uses fake_spectra's single
    redshift with a linear Hubble flow.
- **Precision.** SkewerJAX follows `jax_enable_x64` and never sets it. Every validated tolerance is a float64 one.
- **Purity.** Every container is a JAX pytree, and every kernel can be traced with `jit`, `vmap` and `grad`. The
  host-side parts are NumPy: snapshot I/O, the k-d tree selection, the deposition bookkeeping and fake_spectra.

## `skewerjax.core`

| task | entry point | main options |
|---|---|---|
| lines | `LineBundle(["HI-LyA", "SiIII-1206"], ref_line_id=...)` | any id of `core/lines_table.dat` (`LINES`) |
| background | `Cosmology(H0, Omega_r_0, Omega_m_0, Omega_k_0, Omega_lambda_0)` | batches of cosmologies; `solve_a_of_x(rtol, atol)` |
| skewers | `SkewerBundle(spatial_points, cosmologies, a0, x0, ...)` | `ICs_defined_at_index`, `ICs_at_emission_or_absorption`, `manual_scale_factors` |
| fields | `Fields({species: {name: ScalarField/VectorField}})` | samplers `LogRamp`, `VelocityRamp`, `Constant`, `GRFDensity` (n = n̄(1 + δ) of a `PowerLawGRF`), `SampledField` |
| τ | `compute_optical_depth(bundle, v, lines, fields, a_dagger=...)` or `compute_optical_depth(pieces, v, lines)` | `subdivide`, `line_frame`, `velocity_period`, `voigt`, `detuning`, `resolution_tolerance`, `v_chunk`, `piece_chunk`, `batch_size`, `per_piece`, `dtype`, `devices` |
| τ (layers) | `pieces_from_fields` → `Pieces` → `optical_depth_from_pieces` → `optical_depth_kernel` | `Pieces` is the boundary with `skewerjax.simulations`; `resolution_report(pieces, lines)` |
| frames | `compute_optical_depth(..., line_frame="reference")`, `wrap_to_ref_line`, `velocities_in_line_frames`, `line_wavelength_grid_A` | |
| overflux | `DeltaFieldBuilder(tau, labels=...).build()` | every line subset and the exact decomposition of the total |
| statistics | `OverfluxStatistics1D(ov, dv).compute_p1d().compute_xi1d()`, `OverfluxStatistics3D` | pairs, orders, binning (`bin_p1d`, `bin_p3d_kperp`) |
| metal models | `fit_mcdonald_a`, `fit_ma_model_profiled` | |

## `skewerjax.simulations`

```python
from skewerjax.core import LineBundle
from skewerjax.simulations import ArepoHDF5Reader, SimulationSkewers, ExtractionOptions, IntegrationOptions, HI

reader = ArepoHDF5Reader("/home/STORAGE/TNG50-4", 25)
sk = SimulationSkewers.axis_aligned(reader, [[17500.25, 17500.75, 0.0]], [3], spacing_kms=0.25)  # cofm in ckpc/h, axis z
v, tau = sk.optical_depth(LineBundle(["HI-LyA", "SiIII-1206"], ref_line_id="HI-LyA"))           # (2, 1, N_v)
```

- **Skewers:** `SimulationSkewers.axis_aligned` (straight, along a box axis) or `SimulationSkewers.from_bundle` (any
  `SkewerBundle`, with its own a(x)). The constructor selects the cells near the paths, streaming the snapshot in
  blocks of `max_rows` cells. Host memory is therefore set by the block size, not by the snapshot size.
- **Physics options** (`ExtractionOptions`): `deposition` (`"voronoi"` default, `"tophat"`, or the opt-in
  `"voronoi_crossings"`: one absorber per sub-piece of each cell crossing, split so that its velocity width stays below
  0.25 b), `mass_unit`, `T_constants`, `b_constants`, `x_HI_model`, `cloudy_density`, `cloudy_interpolation`.
- **Selection back ends:** `selection.select_cells_near_paths(..., backend="cpu" | "jztree" | "auto")` and
  `SimulationSkewers(..., selection_backend=...)`: `"cpu"` (default, SciPy's periodic `cKDTree`); `"jztree"` (CUDA GPU,
  needs the optional package jztree; borderline cells re-checked in float64 so the set is the CPU one); `"auto"` (the
  GPU when jztree and a CUDA device are present, else the CPU).
- **Deposition back ends:** `deposition.voronoi_segments(..., backend="jax")` computes the same Voronoi pieces in JAX
  with static shapes (identical cells and lengths; slower on a CPU, so the NumPy path is the default).
  `SimulationSkewers.crossings()` keeps the cell crossings of the skewers alongside their points (`CellCrossings`).
- **Integration options** (`IntegrationOptions`): `frame`, `normalisation`, `detuning`, `dtype`. Periodicity is not one
  of them: `sk.tau(..., velocity_period=...)` and `sk.optical_depth(..., velocity_period=..., line_frame=...)` pass the
  core's option through (default: none). `spectral_grid(res, frame)` spans the velocity span of the box and implies no
  wrap.
- **Lower layers:** `sk.absorbers(ion)` gives the deposited pieces, `sk.pieces(ion, lines)` the same as `core.Pieces`
  (for `compute_optical_depth`), `tau_absorbers(A, lines, v, frame)` the τ (options `velocity_period`, `chunk`,
  `v_chunk`, `dtype`, `devices`, `skewers`), and `sk.tau(...)` the τ cached in an HDF5 file when the object has a `name`
  and a `cache`. `select_cells_near_paths` also takes `GasCells` already in memory (e.g. toy cells, notebook 05).
- **Figure tools:**
  - `slab_projection(reader, normal_axis, centre, thickness, n_pix)` gives the gas surface density and the
    mass-weighted temperature of a slab;
  - `tau_by_path_bins(A, lines, v, frame, s_edges)` gives the τ made by each stretch of the path
    (`00_quickstart_step_by_step`).

## `skewerjax.comparison`

- **Modes:** `FAKE_SPECTRA_EXTRACTION` + `FAKE_SPECTRA_INTEGRATION` reproduce fake_spectra (the matched mode), and
  `ExtractionOptions()` + `DEFAULT_INTEGRATION` are SkewerJAX's defaults. Both use `velocity_period=True`, because
  fake_spectra's spectra are periodic by construction. `matched_tau(sk, ion, lines, v, ext, integ)` evaluates any
  combination.
- **Effect chain:** `run_effects` and `effect_rows` switch one difference at a time (`EXTRACTION_EFFECTS`,
  `INTEGRATION_EFFECTS`).
- **fake_spectra itself:** `FakeSpectraSource(cache=DIR, snap_base=..., snapnum=...)` runs it live, in its own process
  (`python -m skewerjax.comparison.driver full ...`). `FakeSpectraSource(reference=FILE)` reads a file exported earlier
  with `FS.export(FILE)`.
- **Benchmark:** `comparison.benchmark` (`make_config`, `run_suite`, `load_results`, `summary`); see "Benchmarks" below.

## Speed and memory

| knob | where | effect |
|---|---|---|
| `v_chunk` | `compute_optical_depth`, `optical_depth_from_pieces`, `tau_absorbers` (default 4096) | velocities per kernel call: memory ∝ N_lines × N_pieces × v_chunk |
| `batch_size` | `compute_optical_depth`, `optical_depth_from_pieces` | skewers vmapped at once inside `lax.map` (more memory, fewer launches) |
| `piece_chunk` / `chunk` | `compute_optical_depth` / `tau_absorbers` (default 2048) | pieces per kernel call: each skewer's pieces summed in rows (`Pieces.rows`; a static shape) |
| `dtype="float32"` | `compute_optical_depth`, `tau_absorbers`, `IntegrationOptions` | opt-in: 2.4× faster on 16 TNG50-4 skewers on the CPU, max \|ΔF\| 1.6e-3 |
| `devices=` | `compute_optical_depth`, `tau_absorbers`, `optical_depth_sharded` | shards the skewer axis (or a skewer's absorber rows) over devices |
| `max_rows` | `SimulationSkewers` | cells per streamed block of the selection |
| `deposition="voronoi_crossings"` | `ExtractionOptions` | one absorber per (sub-piece of a) cell crossing instead of per piece of every dual segment: ~10× fewer H I absorbers at 0.25 km/s spacing, independent of the spacing |

Sharding uses one call for any number of devices:

```python
import jax
from skewerjax.core import optical_depth_sharded
tau = optical_depth_sharded(pieces, v, lines, devices=jax.devices())    # = optical_depth_from_pieces(...) bit for bit
```

The skewers are padded to a multiple of the device count, and the padding is dropped from the result. On one device the
call falls back to `optical_depth_from_pieces`. To try several devices on one CPU, start Python with
`XLA_FLAGS=--xla_force_host_platform_device_count=N`. This tests the code path only; XLA already uses all the cores of a
single CPU device.

## Validation, tests and benchmarks

```bash
python scripts/validation/run_all.py [--quick] [--skip tng,fake] [--outdir validation_output]
pytest                                                                  # tests/, CPU, seconds per test
SKEWERJAX_SLOW=1 pytest                                                 # also the TNG50-4 / fake_spectra checks
python scripts/benchmarks/benchmark_vs_fake_spectra.py --sets quick --results benchmark_results/suite
python scripts/benchmarks/single_cpu_modes.py --results benchmark_results/single_cpu    # one core, every integration mode
```

The benchmark runs every configuration in its own process with its own CPU set and thread count. It skips the
configurations whose result already exists, and it caps each one at `--cap` seconds. The notebook
`08_fake_spectra_comparison` (Part 2) only reads the results (`RESULTS_DIR`).

## Figures: `skewerjax.utils.plotting`

One call at the top of a notebook or script applies the style of every figure:

```python
from skewerjax.utils.plotting import use_style, use_backend
use_backend()          # interactive figures (ipympl, "%matplotlib widget") in a live notebook; inline when headless
use_style()            # "paper" (default): white background, publication look, Paul Tol's colour-blind-safe "bright" scheme
use_style("dark")      # black background, NeonGlow under every line (the same colours, in the same order)
from skewerjax.utils.plotting import ENTITY_COLORS as EC, PALETTE, CMAPS, FG, BG, DIM, LABELS   # FG, BG, DIM follow the style
```

- Only the background, the foreground (`FG`, `BG`, `DIM`, `ENTITY_COLORS["total"]`) and the direction of the
  sequential maps (`CMAPS["sequential"]`, `"hot"`, `"line"`: low values merge with the background) change between the
  two styles. Import `FG`, `BG` after `use_style`.

- Colours: every discrete colour is Tol's "bright" scheme (`PALETTE` = `tol_colors.colorsets["bright"]`, vendored with
  identical values when `tol_colors` is not installed), in Tol's order (blue, red, green, yellow, cyan, purple, grey).
  `ENTITY_COLORS` fixes the colour of every line and quantity in every notebook (Lyα blue, Si III red, density green,
  temperature purple, velocity yellow, ...). Continuous maps are Tol's (`CMAPS`, registered as `tol_<name>[_r]`).
- Text: minimum font sizes `FONT_MIN` (ticks 16, labels 20, legends / annotations / text boxes 15, floor 13);
  `check_fonts(fig)` lists any text below the floor.
- `SKEWERJAX_HEADLESS=1` forces the inline backend (e.g. when executing notebooks with nbconvert).

- Line styles per mode: `MODE_STYLES`; labels with units: `LABELS`; sizes: `FIGSIZES`.
- `plot_dag(highlight, external=False)`: the dependency graph of the ingredients (`DAG_NODES`, `DAG_EDGES`).
- Path geometry: `geometry_style(name)`, the one style of points, midpoints, dual segments, tangents, line elements,
  piece ends and generators (03 and 05).
- Tutorial figures: `plot_line_table`, `plot_scale_factor_solutions`, `plot_skewer_geometry_3d`,
  `plot_isotropic_recovery` (+ `correlation_binned`), `plot_kpar_kperp_maps`, `plot_pk_mu_wedges` (`MU_WEDGES`, + `power_kmu`,
  `field_power_modes`, `binned_power`, `power_map_kpar_kperp`), `plot_nH_T_fractions`, `plot_deposition_steps`
  (+ `cell_colors`, `voronoi_polygons_2d`), `plot_voigt_limits`, `plot_frames_axes`, `plot_periodic_pieces`,
  `plot_effect_fluxes`, `plot_paths_3d`, `plot_skewer_anatomy`, `plot_voronoi_discretisations`,
  `plot_line_frames`, `plot_flux_vs_analytic`, `plot_terms_by_order`,
  `plot_p1d_input_vs_measured` (+ `grf_p1d_along_axis`), `plot_cylinder(..., progress=p)` (one frame per call),
  `plot_slab`, `plot_tau_buildup`, `plot_orders_superimposed`, `plot_pk_recovery`, the benchmark figures `plot_bench_*`.
- `skewerjax.utils.tutorial`: the small toy set-ups of the notebooks (`toy_pieces`, `periodic_pieces_demo`) and the
  expected spectra of a GRF along skewers (`p1d_expected`, `p3d_expected_on_skewer_grid`,
  `binned_input_power`).
