# Changelog

All notable changes to SkewerJAX. Dates are those of the working sessions; the package version is still 0.0.1
(no release has been tagged).

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
