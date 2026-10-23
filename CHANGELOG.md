# Changelog

All notable changes to SkewerJAX. Dates are those of the working sessions; the package version is still 0.0.1
(no release has been tagged).

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
