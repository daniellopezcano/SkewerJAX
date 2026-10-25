# Changelog

All notable changes to SkewerJAX. Dates are those of the working sessions; the package version is still 0.0.1
(no release has been tagged).

## Unreleased — one field representation: Voronoi cells for every source, one deposition route to τ (2026-10-25, session 20261025_0015)

### Added
- **`core.Cells`** (new module `core/cells.py`): the one container of the gas, for toy models and simulations alike.
  Generating points [cMpc], volumes [cMpc³] (size V^(1/3)), the periodic box, optional ids, and per species
  `number_density` [comoving cm⁻³], `temperature` [K], `velocity` [(M, 3), km/s], optional `turbulence` [m/s] and
  `collisional` [GHz] (zero when left out). A pytree; `take`, `select_species`, `merge_species`, `get`, `size_cMpc`;
  `grid` = the shape of a regular grid of cells. Builders: `cells_from_values` (values as scalars, arrays, functions of
  the generating points, or per-species dicts), `cells_on_grid`, `grid_points`, `voronoi_volumes` (Monte Carlo
  Voronoi volumes in a periodic box).
- **`core.grf_cells(delta, grf, n_bar, generators=None, temperature=, velocity=, ...)`**: the cells of a `PowerLawGRF`
  realisation, n = n̄(1 + δ) with no clipping. Default generators: the grid nodes (the cells are the grid cells: an exact
  GRF at grid resolution); or any generating points (e.g. random), each taking δ interpolated trilinearly at its
  position (`core.interpolate_periodic`), with its Voronoi volume.
- **The route cells → pieces in the core:** `core.select_cells` (new `core/selection.py`, the selection rule and its CPU
  / JZ-TREE back ends), `core.deposition` (moved from `simulations`, plus `grid_segments`, the exact fast path for
  regular grids of cells, tested against `voronoi_segments` in `tests/test_cells.py`), `core.absorbers` (moved from
  `simulations`: `Absorbers`, `ExactFrame`, `SnapshotFrame`, `absorber_pieces`, `tau_absorbers`, ...), and the one
  entry point `core.cell_pieces(cells, lines, skewer_bundle= | points= + frame=, deposition=...)` with `deposit` and
  `bundle_frame`. `ExactFrame(photon_sign=, a_dagger_per_skewer=)`.
- `SimulationSkewers.to_cells(species, ext, override)`: the selected cells after the gas physics as `core.Cells`
  (n_ion a_snap³, T, peculiar velocity); `SimulationSkewers.pieces` goes through `core.cell_pieces`.
- `utils.plotting`: `correlation_binned` returns the exact Gaussian covariance of the binned ξ (and `chi2_cov`);
  `plot_isotropic_recovery` uses it (error bars, the variance in the common mode of the bins, χ² with the covariance);
  `MU_COLORS`. `utils.tutorial`: `cell_window` (sinc² of piecewise-constant cells), `window=` in `p1d_expected` and
  `p3d_expected_on_skewer_grid` (replaces `interpolated=`).

### Changed (breaking)
- `core.compute_optical_depth(bundle, v, lines, cells, *, a_dagger, ii_cosmology, deposition, selection_backend, ...)`:
  a `SkewerBundle` source takes `Cells` (the pieces of `cell_pieces`); `fields=` and `subdivide=` are gone.
- `core.line_profiles(v, lines, *, temperature, los_velocity, scale_factor, a_dagger, turbulence, collisional, ...)`:
  profiles of gas elements given by arrays (no `Fields`).
- `Absorbers.n` is the comoving density (it was the physical density at `a_data`); `a_data` is optional;
  `absorbers_from_components(seg, points, cells, species, mass_amu, a_data=)` and `absorbers_from_crossings(...)` read
  the cells; `Absorbers` carries optional `turbulence` / `collisional`; `column_cm2(a=None)`.
- Moved, no aliases: `skewerjax.simulations.deposition` → `skewerjax.core.deposition`; `skewerjax.simulations.absorbers`
  → `skewerjax.core.absorbers`; the selection helpers to `skewerjax.core.selection` (`simulations.select_cells_near_paths`
  streams a snapshot and calls `core.select_cells` for cells in memory). `skewerjax.simulations` no longer re-exports
  `Absorbers`, `ExactFrame`, `SnapshotFrame`, `absorber_pieces`, `absorbers_from_*`, `sampling_check`, `tau_absorbers`,
  `tau_by_path_bins`, `CellCrossings`, `SegmentComponents`, `cell_crossings`, `tophat_segments`, `voronoi_segments`.
- `plot_kpar_kperp_maps` has no residual row (returns axes (n_cases,)); `plot_pk_mu_wedges` draws one panel per case
  with every wedge and its residual panel below; DAG labels: toy models / simulation outputs → gas cells.

### Removed
- The route of fields known everywhere: `core.fields` (`Fields`, `ScalarField`, `VectorField`, `SampledField`,
  `tile_field_to_lines`), the samplers `GRFDensity`, `LogRamp`, `VelocityRamp`, `Constant`, `core.pieces_from_fields`,
  the `subdivide` option, `optical_depth._line_fields`, `SimulationSkewers.fields`. Replacements: `Cells` and its
  builders, `grf_cells`, `cell_pieces`, `compute_optical_depth(bundle, v, lines, cells)`.

### Numerical changes
- Simulation route: unchanged (bit-identical τ for the default Voronoi and the crossings deposition; the top-hat option
  differs by ≤ 2e-16 relative, the order of the multiplications by 3/4π and a³).
- Toy route: the gas is piecewise constant per cell instead of trilinearly interpolated (intended). Notebook 00's box:
  ⟨F⟩ +1.8e-4, max |ΔF| 6.2e-2; the P1D agrees at low k (1 %) and carries the window sinc² of the cells instead of the
  sinc⁴ of linear interpolation at high k. The analytic checks are unchanged (identical numbers, same tolerances).

### Notebooks
- 04: GRFs only (ramps and the pytree section removed); ξ with its exact covariance; maps without residuals; wedges in
  one panel per case; new closing section: from the GRF to cells (grid and random generators, the deposited P1D).
- 05: ends with the snapshot as `core.Cells`; the step-by-step selection → deposition → τ moves to 06.
- 06: opens with that step-by-step section on `Cells`; §1 states pieces from cells only; the convergence block of fields
  known everywhere is removed; §3–§5 and §7 use the cell route.
- 07: toy spectra on GRF → cells → τ; the expected projections carry the window of the cells.
- 00, 00_quickstart_step_by_step: the new route (printed results unchanged).

## Unreleased — fifth pass: one piece-based τ route, opt-in periodicity, notebooks 03–07 (2026-10-24, session 20261024_0040)

### Changed (breaking)
- **One route for τ: pieces.** `core.Pieces` (replaces `LoSSamples`) holds the constant-property pieces of every skewer,
  each at its centroid, plus `width_kms` (the line-centre variation across the piece) and `velocity_span_kms`.
  - Fields known everywhere: `core.pieces_from_fields(bundle, fields, lines, subdivide=1)` (replaces `los_samples`):
    one piece per dual segment (or `subdivide` equal ones), with a, the position, the photon direction and every field
    at the piece's centroid (a interpolated in ln a between the bundle's points). The `line_elements=` option is gone.
  - Simulation outputs: `simulations.absorber_pieces(ab, frame, n_lines, skewers=None)` (replaces `absorber_samples`);
    `SimulationSkewers.pieces(ion, lines)`; `tau_absorbers` now calls `core.compute_optical_depth`.
  - `core.compute_optical_depth(source, v, lines, fields=None, ...)` takes a `SkewerBundle` (with `fields`) or `Pieces`;
    new options `subdivide`, `line_frame`, `piece_chunk`, `per_piece`. `optical_depth_from_samples` →
    `optical_depth_from_pieces`; `optical_depth_kernel(..., length_m, per_piece=)`; `resolution_report(pieces, lines)`
    uses the piece widths (`sampling_resolution_report` removed).
- **Periodicity is one opt-in option of `compute_optical_depth`:** `velocity_period=False` (default, no wrap), `True`
  (each skewer periodic over its own velocity span, set by a at its first and last points), or a period [km/s]. It is
  a traced per-skewer array inside the kernel. `SimulationSkewers.tau / optical_depth / compute_tau` and
  `tau_absorbers` pass it through.
  - Removed: `IntegrationOptions.periodic` (the simulation default was periodic; it is now the core's non-periodic
    default), `SimulationSkewers.frame(kind, periodic)` → `frame(kind)`, `ExactFrame/SnapshotFrame.period_kms`
    (→ `velocity_span(skewer)`), the buffer (`SimulationSkewers.axis_aligned(..., extend_cMpc=)`, `s_origin`,
    `axis_aligned_skewer_points(extend=)`, `column_per_pixel(s_origin=)`).
  - `SnapshotFrame(a_snap, velfac, path_length)`. `spectral_grid(res, frame)` spans the velocity span and implies no wrap.
  - The comparison layer sets `velocity_period=True` explicitly (`FAKE_SPECTRA_INTEGRATION`, `DEFAULT_INTEGRATION`,
    `MatchedIntegration.velocity_period`, the benchmark), because fake_spectra is periodic by construction.
- **Frames are not periodicity.** `compute_optical_depth(..., line_frame="reference")` evaluates every line on the
  reference line's axis (line j at v + Δv_j), with no interpolation. `wrap_to_ref_line(tau, v, lines, *,
  velocity_period=None, fill_value=nan)` no longer wraps by default: NaN where the shift leaves the grid; it wraps
  modulo P only with `velocity_period=P` (a grid of one period). `spectral.frequency_offset_GHz` takes per-line grids.
- Removed the point-valued deposition (`deposition="points"`, `tophat_point_values`, `absorbers_from_point_values`):
  it sampled the cells at the points, not over the pieces.
- `SampledField` repeats each point's value over the pieces of its dual segment.

### Added
- `select_cells_near_paths` accepts `GasCells` already in memory (toy cells; the same selection rule).
- `simulations.absorbers_from_components` exported.
- Plotting: `geometry_style` (one style for points, midpoints, dual segments, tangents, line elements, piece ends,
  generators); `plot_isotropic_recovery` (+ `correlation_binned`), `plot_kpar_kperp_maps` (residuals, χ²/dof),
  `plot_pk_mu_wedges` (`MU_WEDGES`, `power_kmu`); `plot_deposition_steps` (+ `cell_colors`, `voronoi_polygons_2d`);
  `plot_periodic_pieces`. `plot_skewer_geometry_3d(scale_factors=None)` no longer colours by a(x).
  `plot_frames_axes(..., v_ref=, F_ref=)`: own frame, global axis, observed wavelength.
- `utils.tutorial`: `toy_pieces`, `periodic_pieces_demo`.
- Tests: `tests/test_pieces.py`; updated `test_wrap_to_ref_line`, `test_precision_and_sharding`, `test_tutorial`,
  `test_plotting`, `test_deposition`.

### Removed
- `plot_pk_recovery_nd`, `plot_deposition_story`, `plot_two_elements`, `plot_linear_detuning`,
  `plot_periodicity_options`; `tutorial.two_elements`, `linear_detuning`, `periodic_box_demo`.

### Numbers (against the committed code, 14f6b3f)
- TNG50-4 (6 configurations, 1 and 16 skewers, Voronoi and crossings, H I and Si III) with `velocity_period=True`:
  bit-identical to the committed periodic default.
- New default (no wrap) on the 16 TNG50-4 skewers: ⟨F⟩ +6.6e-3, max |ΔF| 0.998 at an edge, P1D up to 25 % at
  k < 0.1 s/km; on one skewer ⟨F⟩ +1.5e-2.
- Toy route, dual-segment centroids: interior pieces of even straight skewers unchanged to rounding (5e-5 of τ_max with
  float32 input points); only the end pieces move (by a quarter of the spacing). Uneven helix: the new route is closer
  to the converged (subdivided) τ (1.6e-2 vs 2.8e-2 of τ_max for Lyα).
- `check_optical_depth_analytic`: the radiation-domination errors fall by 2–4× at every resolution.

### Notebooks
- 03: the 3D geometry figure without a(x) colouring, each element in its own style. 04: block A (P(k), ξ(r)),
  block B ((k∥, k⊥) maps with residuals; P(k) in μ wedges). 05 §3: built from the package's classes on toy cells.
  06: §1 around pieces, two-element block and linear detuning removed, §5 frames only, §6 periodicity under the new
  option, §7 simulation outputs. 07: the deleted first block of §1 restored; periodic spectra (FFT estimators) and one
  axis for both lines. 00, 00b: the same choices where their spectra are used as periodic. 08: one call
  (`sk.tau(..., velocity_period=True)`, like-for-like with fake_spectra).

## Unreleased — notebooks fourth pass: paper/dark styles, τ derivation, periodicity, GRF study, selection back ends (2026-10-23, session 20261023_0530)

### Added
- `simulations.selection.select_cells_near_paths(..., backend="cpu" | "jztree" | "auto")`, `jztree_available`,
  `resolve_backend`, and `SimulationSkewers(..., selection_backend=...)`:
  - `"cpu"` is the default (unchanged);
  - `"jztree"` is the CUDA back end through the optional package jztree, with borderline cells re-checked in float64;
  - `"auto"` uses the GPU when jztree and a CUDA device are present.
  - jztree is not installed in the development environment (its wheels need jax ≥ 0.8 and NumPy ≥ 2.2.6), so the GPU
    path is untested and `"auto"` resolves to the CPU.
- `core.GRFDensity(key, grf, n_bar, delta_grid=None)`: n = n̄(1 + δ) from one `PowerLawGRF` realisation, trilinear in
  the periodic box, with no transform; `min_one_plus_delta`, and a warning when it is ≤ 0.
- `skewerjax.utils.tutorial`: the toy set-ups of the notebooks:
  - `two_elements`, `linear_detuning`, `periodic_box_demo`;
  - the expected spectra of a GRF along skewers: `p1d_expected`, `p3d_expected_on_skewer_grid`, `binned_input_power`.
- Plotting:
  - `use_style("paper")` (new default: white, publication look) and `use_style("dark")`. Both use the same palette,
    with `FG`, `BG`, `DIM`, `CMAPS` following the style (`THEMES`, `matplotlib_paper_config`).
  - New figures: `plot_skewer_geometry_3d`, `power_map_kpar_kperp`, `plot_two_elements`, `plot_linear_detuning`,
    `plot_periodicity_options`.
  - `plot_cylinder(..., flux_on_cylinder=False)`.
- Tests: `tests/test_selection_backend.py`, `tests/test_toy_fields.py`, `tests/test_tutorial.py`, and new figure and
  style tests in `tests/test_plotting.py`.

### Changed
- **Default plotting style:** paper (white) instead of dark.
- **The dependency graph:**
  - one highlight (a light fill and a coloured edge; no halo);
  - dashed edges with a fixed dash pattern, so that they look dashed in every orientation (the arrow heads no longer
    stroke the last segment solid);
  - highlight lists per notebook = the ingredients each one uses.
- `plot_pk_recovery_nd`: the second row is the measured power in the (k∥, k⊥) plane with input iso-contours.
- `plot_deposition_story` panel (c): the per-piece contributions dτ_p and their sum, as derived in notebook 06.
- `plot_frames_axes`: no arrows in panel (b).
- Notebooks 00–07 rebuilt:
  - markdown maths in LaTeX throughout;
  - no external code named;
  - 03: one 3D figure of one skewer, plus its anatomy;
  - 04: no `GaussianDensity` section; a pytree explanation;
  - 05: no piecewise figure and no sampling section;
  - 06: the derivation from the integral to the sum, the linear-detuning test, periodicity rewritten, and simulation τ
    at the end;
  - 07: a systematic GRF study.

### Removed
- `core.GaussianDensity` (the bounded tanh transform). Use `GRFDensity`.
- Plotting: `plot_path_elements`, `plot_path_elements_3d` (replaced by `plot_skewer_geometry_3d`), `plot_option_sweep`,
  `plot_raw_to_overflux` (no longer used).

## Unreleased — notebooks third pass: clarity, consistency, two code items (2026-10-22, session 20261022_2120)

### Added
- `simulations.deposition.voronoi_segments(..., backend="jax")`: the Voronoi deposition in JAX with static shapes
  (chunked candidate search with a provable completeness test, the same exact crossings, sort + segment-sum merging).
  Identical cells and bit-identical lengths to the NumPy default on TNG50-4; slower on a CPU, so opt-in.
- Voronoi-adapted segmentation: `deposition.CellCrossings` / `cell_crossings`, `SimulationSkewers.crossings()` (kept
  alongside the points), `absorbers.absorbers_from_crossings` and `ExtractionOptions(deposition="voronoi_crossings")`
  (opt-in): one absorber per sub-piece of each cell crossing, split so that W <= 0.25 b. The column is exact for any
  split; τ converges to the default as the tolerance falls.
- `scripts/benchmarks/single_cpu_modes.py`: one-core timing of SkewerJAX's modes on the inputs of the fake_spectra
  benchmark; `notebooks/benchmark_records/` (the saved benchmark records read by notebook 08).
- Plotting: `use_backend()` (ipympl in a live notebook, inline when headless), `FONT_MIN` and `check_fonts`, `CMAPS`
  (Tol's maps registered as `tol_<name>[_r]`), `LINE_COLOR_ORDER`; figures `plot_scale_factor_solutions`,
  `plot_path_elements`, `plot_path_elements_3d`, `plot_pk_recovery_nd`, `field_power_modes`, `binned_power`,
  `plot_nH_T_fractions`, `plot_deposition_story`, `plot_voigt_limits`, `plot_frames_axes`, `plot_option_sweep`,
  `plot_effect_fluxes`; `plot_orders_superimposed` marks k = 2πn/Δv on P1D.
- Tests: `tests/test_deposition.py` (JAX back end vs NumPy, cell crossings), style checks in `tests/test_plotting.py`.

### Changed
- Notebooks restructured: `00_quickstart_step_by_step` (from `09_visualisations`, now a step-by-step story with the
  equations), `04_fields_and_samplers` (was 05), `05_simulation_outputs` (was 04); the summary sections are gone
  (except in 08). New content per notebook: see the reply of the session.
- Style: every discrete colour from Tol's "bright" scheme (`tol_colors`), a fixed colour per line and quantity,
  Tol's continuous maps, minimum font sizes; the dependency diagram redrawn (field sources side by side, labelled
  edges, a legend, no overlapping arrows).
- Docstrings, scripts and tests no longer cite note file names or pre-refactor notebooks.

## Unreleased — notebooks second pass: dependency graph, one dark style, pedagogical content (2026-10-22, session 20261022_1520)

### Changed
- **Notebooks renumbered along the dependency graph of the ingredients** (derived from the code): `00_quickstart`,
  `01_lines`, `02_cosmology`, `03_skewers`, `04_simulation_outputs`, `05_fields_and_samplers`,
  `06_line_profiles_and_optical_depth`, `07_summary_statistics`, `08_fake_spectra_comparison` (the former 06 and 08
  merged), `09_visualisations`. The former `01_fields_and_samplers`, `02_cosmology_and_skewers`,
  `03_line_profiles_and_optical_depth`, `04_summary_statistics`, `05_tng_extraction`, `06_fake_spectra_comparison`,
  `07_visualisations` and `08_benchmark_skewerjax_vs_fake_spectra` are removed (their content moved). The τ sections
  of the former 05 are now section 7 of 06.
- **One style for every figure.** `use_style()` is dark by default: black background, a glow under every line
  (`NeonGlow`, a path effect) and Paul Tol's colour-blind-safe "light" scheme (`PALETTE`), which stays luminous on
  black. `ENTITY_COLORS`, `NEON_COLORS` and the benchmark colours use it; `use_style("light")` keeps the old style.
  Helpers read the foreground colour from the style (`FG`, `BG`).
- The "Block A/B/C" labels are gone from the diagrams and texts of the notebooks, `README.md` and `docs/`.

### Added
- `plot_dag` (`DAG_NODES`, `DAG_EDGES`): the dependency graph that opens every notebook, with its ingredients
  highlighted, and the lines "Requires" / "Independent of" in every introduction. It replaces `plot_pipeline`
  (removed, with `PIPELINE_STEPS` and `PIPELINE_SIDE`).
- Figure functions for the new sections: `plot_line_table` (01), `plot_paths_3d`, `plot_skewer_anatomy` (03),
  `plot_voronoi_discretisations` (04), `plot_line_frames`, `plot_flux_vs_analytic` (06), `plot_raw_to_overflux`,
  `plot_terms_by_order`, `grf_p1d_along_axis`, `plot_p1d_input_vs_measured` (07), `plot_cylinder` (09; `progress=`
  draws one frame of a future animation), `dark_3d`, `box_edges_3d`, `species_color`.
- New notebook sections: the line classes (01); 3D paths and a bundle's attributes (03); the step-by-step extraction
  with formulas, the three grids on a Voronoi mesh and the column against the sampling (04); line-centred and global
  frames against the analytic profile and the radiation-dominated closed form (06); from the raw τ to δ_F with the
  periodic treatment, P1D/ξ1D by term, input vs measured P1D (07); the differences with fake_spectra step by step and
  the per-pixel column (08); the gas cylinder (09).
- No numerical method changed.

## Unreleased — performance, sharding, benchmark, visualisations, documentation (2026-10-21, session 20261021_2330)

### Added
- **Opt-in single precision for the optical depth.** `dtype="float32"` in `core.optical_depth_from_samples`,
  `core.compute_optical_depth`, `simulations.tau_absorbers` and `simulations.IntegrationOptions(dtype=...)`. The result
  is cast back to the input precision. The default (`None`) is unchanged and bit-identical to before. On 16 TNG50-4
  skewers (CPU) it is 2.36x faster, with max |ΔF| = 1.6e-3 and |Δ<F>| = 2.6e-7 against float64.
- **Sharding over skewers.** `core.optical_depth_sharded` (`shard_map` over a 1-D device mesh; the skewers are
  edge-padded to a multiple of the device count), and a `devices=` argument in `core.compute_optical_depth` and
  `simulations.tau_absorbers`. One API serves 1 and N devices. It is bit-identical to the unsharded path on CPU ×1,
  CPU ×4 (forced host devices) and one GPU.
- **Benchmark against fake_spectra.** `skewerjax.comparison.benchmark` holds the stage timers, the per-configuration
  worker processes, the agreement check and the result loaders. `scripts/benchmarks/benchmark_vs_fake_spectra.py` is the
  launcher, and the notebook `08_benchmark_skewerjax_vs_fake_spectra` plots the results.
- **Visualisation tools.** `simulations.slab_projection` (gas surface density and mass-weighted temperature of a slab)
  and `simulations.tau_by_path_bins` (τ of each path interval; the intervals add up to the total).
  `simulations.tau_absorbers` gains `skewers=` (a subset of skewers).
- **Plotting module.**
  - Shared quantity labels (`LABELS`) and figure sizes (`FIGSIZES`).
  - A pipeline schematic (`plot_pipeline`).
  - Spectra, profile and zoom-inset helpers.
  - The dark "neon" style of the earlier visualisation notebook (`NEON_COLORS`, `dark_axes`, `neon_line`).
  - Slab and absorption build-up views.
  - Benchmark figures (`plot_bench_*`).
- **Notebooks** `07_visualisations` and `08_benchmark_skewerjax_vs_fake_spectra`. Notebooks 00–06 open with the pipeline
  schematic and share one structure.
- **Documentation:** `docs/installation.md`, `docs/usage.md`, a rewritten `README.md` and this changelog.
- **Tests:** a pytest suite in `tests/` (CPU, toy inputs).

### Changed
- `pyproject.toml`:
  - complete metadata;
  - dependencies with lower bounds taken from the validated environment;
  - extras `gpu`, `gpu-local`, `comparison`, `notebooks` and `dev`;
  - package data;
  - `requires-python >= 3.11` (scipy 1.16 requires it).
- `tests/test_wrap_to_ref_line.py` is ported to the current API. It had failed at import since the refactor below.
- `.gitignore`: JAX/XLA caches and the output folders of the validation and benchmark scripts.

### Not changed
- Physics and numerics with default options: the 800 baseline quantities of the validator are bit-identical.

## 2026-10-21 — refactor into three blocks (session 20261021_2130, commit e0caf7a)

### Changed
- `skewerjax/` was rewritten in place into three one-way blocks, JAX-first:
  - `core` (lines, cosmology, skewers, fields, Voigt profiles, optical depth, overflux, statistics);
  - `simulations` (snapshot reading, gas physics, cell selection, deposition, absorbers, `SimulationSkewers`);
  - `comparison` (fake_spectra's per-cell scheme, the matched options, the effect chain, the fake_spectra driver).
- The package shrank from 13,358 to 6,128 lines. Results stay bit-identical to the previous version for 797 of 800
  validated quantities, and the other 3 agree to 3e-11.
- Containers are JAX pytrees. The per-skewer Python loop of the optical depth is replaced by `lax.map`.
- The notebooks were replaced by seven tutorials (`00`–`06`). Validation moved to `scripts/validation/` (nine
  `check_*.py` scripts and `run_all.py`).
