# Changelog

All notable changes to SkewerJAX. Dates are those of the working sessions; the package version is still 0.0.1
(no release has been tagged).

## Unreleased — readable velocity arrows in every figure (2026-10-26)

Presentation only: no result changes.

### Changed
- One rule for the velocity arrows of `plot_gas_cells` (04, 05) and the deposition figures (07):
  `utils.plotting.velocity_arrow_scale` sets the length per km/s from the velocities a figure draws (the
  90th-percentile speed is 0.8 x the median size of the cells shown, and at least 6 % of the panel's largest span),
  one scale per figure (`plot_deposition_pieces` and `plot_deposition_rings` share one from the v_par they draw;
  `plot_deposition_cells` its own from |v|). The override stays (`arrow_scale=` / `arrow_kms=` + `arrow_cMpc=`); the
  scale goes to `info["arrow_scale"]`.
- Every arrow stays inside its panel (`_fit_arrows`): the limits grow by up to 10 % of the span per side, and an
  arrow still too long is cut at the edge and ends with an open circle (said in the legend / title).
- A reference arrow of a round speed (`round_speed`, `velocity_key_speed`: the median speed rounded to 1, 2 or
  5 x 10^n km/s) replaces the "arrows: ... km/s per ... cMpc" text: in 3D panels inside the scene
  (`_velocity_key_3d`: along the axis most parallel to the screen, at the clearest upper corner of the box in
  projection, so the projection shortens it like the arrows; its label past the tip or beside it, wherever clearest;
  "lengths in projection" in the title or the legend), in
  2D panels in a small box (`_velocity_key_2d`, and on the ring of (c)).

## Unreleased — repository reorganisation: one test suite, packaging, tooling (2026-10-26)

No result changes, except the one intended: `voigt="convolution"` is no longer a mode of the line profiles and of tau.

### Removed
- `voigt="convolution"` from `compute_optical_depth`, `optical_depth_kernel`, `voigt_line_profiles` and
  `line_profiles`: it returned all-NaN tau (the frequency offsets of tau decrease along an increasing velocity grid,
  and the FFT convolution assumes an increasing uniform grid) and nothing used it. Any mode other than
  `line_profiles.PROFILE_MODES` = ("ExoJAX", "harris") raises a `ValueError` naming them. `core.voigt.voigt_convolution`
  and `log10_voigt(mode="convolution")` stay, as standalone functions (notebook 06).
- `scripts/validation/` (`check_*.py`, `run_all.py`, `_common.py`) and `tests/test_validation_scripts.py`: every check
  is now a test, with the same tolerances.
- `configs/global_setup.yaml` (read by nothing; its float32 / x64-off values contradicted the package),
  `cuda-keyring_1.1-1_all.deb` (moved out of the repository), `SAVED_FIGURES/` (empty), `build/`,
  `skewerjax.egg-info/`, every `__pycache__/`.

### Changed
- **Tests:** one pytest suite laid out like the package: `tests/core/`, `simulations/`, `utils/`, `comparison/`,
  `regression/` (data in `tests/regression/data/`, renamed `route_toy.npz`, `modules_toy.npz`, `modules_tng.npz`,
  contents unchanged). Merged: the two regression modules; streamed selection into `core/test_selection.py`; the frame
  checks into `core/test_spectral_frames.py`; Harris into `core/test_voigt.py`; the precision / sharding tests, the
  periodicity tests and the closed forms into `core/test_optical_depth.py`; the input checks into
  `core/test_validation.py`. Markers `slow`, `tng`, `gpu`, `fake_spectra` (registered in `pyproject.toml`; the
  missing requirements skip the tests with the reason), a fast default `-m "not slow"`, `tests/README.md`.
  `scripts/dev/run_tests.py` and its pytest stand-in support the same markers (for environments without pytest).
- **Benchmarks:** one home, `benchmarks/`: `results/fake_spectra_benchmark.json` (the summary notebook 10 reads, 152
  kB, committed) and `raw/` (the per-run records, git-ignored). `utils.paths` defaults, the benchmark script and
  notebook 10 follow.
- **Packaging:** the core dependencies are what `import skewerjax` needs (h5py moved to the `simulations` extra: it is
  imported where it is used); extras `simulations`, `plots`, `comparison`, `notebooks`, `dev` (pytest, pytest-cov, ruff,
  pre-commit, nbstripout), `gpu`. `.gitmodules` declares the CLOUDY tables submodule (sbird/cloudy_tables, already a
  gitlink without it). `MANIFEST.in` tidied.
- **Tooling:** ruff (`E`, `F`, `I`, `B`, `UP`; line length 130; notebooks excluded) with its fixes applied to the code
  of notebooks 01–07 (imports, annotations, unused names) and imports only in `core/stats` and `comparison`; `ruff
  format` over the package, tests and scripts; `.pre-commit-config.yaml` (ruff, nbstripout `--keep-id`, whitespace,
  end of file, large files); a CI workflow (`.github/workflows/tests.yml`: ruff and the fast CPU tests).
- `.gitignore` rewritten (45 lines instead of the 225-line template).
- New: `scripts/dev/update_regression_references.py`, `scripts/figures/optical_depth_vs_analytic.py`.

## Unreleased — revision checkpoint of notebooks 01–07: input checks, debugging tools, faster streamed selection (2026-10-26)

No result changes: every array of the in-scope code bitwise identical to the previous commit (232 arrays, incl.
TNG50-4), every printed number of the 13 notebooks identical.

### Added
- `core.validation` (internal): eager input checks with errors that name the field and its first bad entry, skipped
  under tracing. Used by `Cells` (new `validate=True`), `SkewerBundle`, `Cosmology`, `LineBundle`, `select_cells`,
  `cell_pieces` and `compute_optical_depth` (`check_pieces`, exported); a one-time warning when `jax_enable_x64` is off.
- `core.diagnostics`: `check_skewers(cells, bundle, lines)` → `SkewerCheck` (coverage, column against a fine
  nearest-generator sampling over all the cells with the missing cells listed, window integral on one period,
  resolution); `enable_logging()`.
- `return_info=True` on `select_cells`, `simulations.select_cells_near_paths`, `deposit`, `cell_pieces` and
  `compute_optical_depth` (counts, lengths, columns, resolution, timings per stage, the window fraction).
- Logging through `logging.getLogger("skewerjax.<module>")` (DEBUG; silent by default, `NullHandler`).
- `voronoi_segments` diagnostic `max_distance_over_size`, with a warning when non-periodic paths leave the gas.
- Tests: `test_robustness.py`, `test_diagnostics.py`, `test_regression_checkpoint.py` (+ `tests/data/regression_checkpoint*.npz`).

### Changed
- `select_cells_near_paths`: the coarse prefilter runs on the code coordinates (one more grid cell of dilation), and
  only its survivors are converted to cMpc, sized and queried; the same cells, faster (see `STATE.md` §7).
- `ArepoHDF5Reader.positions_cMpc` converts in place (one temporary fewer).
- Merged duplicates: one absorber row builder (`absorbers._AbsorberRows`), `core.selection.path_tree` (both
  selections), `simulations.selection.read_selected` (both streamed selections), `snapshot_skewers.ion_cell_fields`
  (`to_cells`, `gas_to_cells`).
- `Cells.select_species` raises `KeyError` naming the missing species.

## Unreleased — notebook 07: the seam figure in three figures (2026-10-26)

### Changed
- `plot_deposition_steps` split into `plot_deposition_cells` (row (a)), `plot_deposition_pieces` (row (b)) and
  `plot_deposition_rings` (rows (c)–(e)), sharing `_deposition_view` (the same pieces, colours and labels in all three).
  The pieces shown are those within `half_range_cMpc` of the seam s = 0 ≡ L (replaces `regions`); the ring of v† shows
  the fixed window ±`half_range_kms` around 0 ≡ P (default 300 km/s), the ring of s the window ±l; each centred on its
  seam, without the grey neighbouring pieces. One velocity-arrow scale in the three figures (`arrow_kms`,
  `arrow_cMpc`: 0.1 cMpc per 100 km/s, in every legend): the 3D cell velocities, and v∥ drawn from the centre of each
  piece along s with its sign, overlapping arrows stacked by `_arrow_levels`. (a): larger panels, no pane grids,
  red tangents, outlined box faces, a row of cell labels; whole rings of (c)–(e) larger, the explanations in one legend.
- `_RingSector(aspect=)` (the depth from the window, so narrow and wide windows fill the same axis), `ticks` (values
  before the seam as L − x with their decimals, the unit on the last tick), `rticks` and `labels` in page units.
- `plot_deposition_rings`: (c)–(e) draw the piece on the velocity seam like every other piece (no bold outline, label
  or dashed profile, no P/2 cut ray, no legend entries). `_density_norm`: one norm of `CMAPS["density"]` (tight log
  range, 0.05 dex pad) for `plot_gas_cells` (04, 05) and the deposition figures (07).
- Notebook 07, §1: the skewer sampled every 20 km/s (`DV_POINTS`, a visualisation choice, not converged), the figure
  cell split in three with `HALF_RANGE_CMPC = 2.0` and `HALF_RANGE_KMS = 300.0`, and the reading of each figure.

## Unreleased — the fake_spectra comparison in two notebooks, and a new benchmark harness (2026-10-26)

### Changed
- Notebook `09_fake_spectra_comparison` split in two: `09_comparison_fake_spectra_results` (what the two codes compute:
  its former Part 1, the same cells with identical outputs and pixel-identical figures; summary numbers from the new
  benchmark; markdown math fixed) and `10_comparison_fake_spectra_performance` (speed and memory: time against N and
  the snapshot size, stages, speed-up, memory, threads, agreement while timed, every record with its preset).
- `comparison.benchmark` rewritten: one process per configuration (pinned CPU set, fixed threads), the first run kept
  apart (compilation, cold read), exclusive stages (`StageTimer`: setup, I/O, selection, physics, pieces, τ), matched
  and default modes in one process on one selection, host RSS per mode (`RssSampler`) and GPU peak, the agreement with
  fake_spectra on up to 64 stored skewers (max and rms |ΔF|), launcher-side memory and time caps, skip prediction
  along a series, `collect`, `result_rows`, `value`, `print_setup`, `print_rows`.
- **Presets** (`PRESETS`, launcher `--preset quick|full`, `--budget`): `quick` (the default) is a reduced study in
  ≤ 30 min of new runs (TNG50-3 at N = 1, 4, 16, TNG50-2 at N = 1, 1–16 threads at N = 1; one or two timed runs; a cap
  and an estimate per point, `QUICK_SCALING`, `QUICK_THREADS`); `full` is the whole study (N up to 1024 on TNG50-4,
  -3, -2; threads at N = 16). `run_suite(total_budget_s=)` starts no point beyond the budget (not written, so the next
  launch measures it); the full preset re-measures quick points with fewer timed runs (`_remeasure`). Every record
  keeps `config["preset"]`; the results file adds `runtime_by_preset_s` and `launches`.
- `snapshot_info` reads only `snap_NNN.*.hdf5` (another HDF5 file in the snapshot folder broke it).
- `utils.plotting`: the benchmark figures `plot_bench_vs`, `plot_bench_speedup`, `plot_bench_stages`,
  `plot_bench_accuracy` (max and rms |ΔF|), `bench_style`, `bench_value`, `bench_slope`; the DAG lists the comparison
  notebooks as 09, 10.

### Removed
- `scripts/benchmarks/benchmark_vs_fake_spectra.py`, `single_cpu_modes.py` (replaced by `benchmark_fake_spectra.py`),
  the old `plot_bench_*` functions, and notebook `09_fake_spectra_comparison`.

### Results (merged: a partial full run, 3.3 h, stopped; and the quick run, 26.5 min)
- TNG50-4: fake_spectra 1.7 s (N = 1) to 16 s (N = 1024); SkewerJAX CPU default 3.0 s to 451 s (N = 256), ≈ 1.8 s per
  skewer; float32 ≈ 2× faster; the GPU slower than the CPU. TNG50-3: SkewerJAX faster than fake_spectra at N = 1
  (15 s vs 32 s), slower from N ≈ 4. TNG50-2: fake_spectra 151 s at N = 1; SkewerJAX over the quick cap (> 600 s for
  the first and one timed run), left for the full preset.
- Matched mode while timed: rms ΔF ≤ 5×10⁻⁶ (Lyα, Si III); max |ΔF| ≤ 6×10⁻⁵ up to N = 4, ~2–3.5×10⁻⁴ at N ≥ 16 from
  one pixel of one skewer per line.

## Unreleased — notebook 07: the seam on rings, Lyα + Si III, `wrap_to_ref_line` panel (2026-10-26, session 20261026_1244)

### Changed
- `plot_deposition_steps(parts, bundle, regions, v, tau_pieces, *, period, ...)` (new signature): the end of the path
  (left) and its start (right) around the seam, 2 cMpc each (a piece straddles the velocity seam). (a) the cells in 3D
  with the box faces (green planes), the skewer (red), unit tangents and peculiar velocities; (b) the pieces, labelled
  by cell and a letter in path order (E3a, E3b, ...), with their v∥ = **v**·t̂ arrows and the box faces; (c) the
  pieces on the ring of s (period L), (d) on the ring of v† at their line centres (period P; the P/2 cut of the seam
  piece in the inset), (e) every piece's dτ_p with `velocity_period=True` and the total τ, on the same ring. The rings
  are drawn on equal-aspect Cartesian axes (`_RingSector`), labels placed by `_spread`, a whole ring per row
  (`_full_ring`).
- `plot_frames_axes(..., wrapped=dict(v, tau, tau_ref, period))`: panel (d), what `wrap_to_ref_line` does to a
  periodic spectrum (each line shifted by −Δv_j modulo P); `figsize` default `None` (by the number of panels).
- `_draw_cells_3d(..., path_color=, arrow_color=)` (default the foreground colour: `plot_gas_cells` unchanged).
- Notebook 07: only Lyα and Si III (`LineBundle(["HI-LyA", "SiIII-1206"], ref_line_id="HI-LyA")`); §1 figure and its
  markdown rewritten (s-ring → v-ring → dτ and τ); §2 adds panel (d) and checks `wrap_to_ref_line` against
  `line_frame="reference"`.

### Removed
- `plot_velocity_ring` and `_label_tracked` (the ring rows of `plot_deposition_steps` replace them), and the
  notebook's separate ring section.

## Unreleased — cleanup, optimisation and reorganisation checkpoint (2026-10-26)

No result changes: every notebook's printed numbers and arrays are identical to the run before the checkpoint, every
figure is pixel-identical, and `tests/test_regression.py` compares the route with arrays of the pre-checkpoint code.

### Removed
- `utils.plotting`: `plot_spectra`, `zoom_inset` (imported by every notebook, never used), `plot_skewer_profiles`,
  `recovery_summary`, `plot_flux_vs_analytic` (unused), `plot_line_frames`, `plot_voronoi_discretisations`,
  `plot_paths_3d` + `box_edges_3d`, `plot_periodic_pieces` (only their smoke tests used them; superseded by
  `plot_frames_axes`, `plot_gas_cells`, `plot_deposition_steps`).
- `utils.tutorial`: `toy_pieces`, `scale_factor_of_x`, `PERIODIC_DEMO_PIECES`, `periodic_pieces_demo` (the two
  periodic effects are tested by `tests/test_pieces.py::test_the_two_periodic_effects`).
- `core.deposit(select=)`, `core.cell_pieces(select=)` (never passed; the selection always runs except on the grid fast
  path). Dead imports in `core/optical_depth.py` and two scripts.
- `tests/test_tutorial.py` (merged into `test_toy_fields.py`).

### Changed
- One wrap helper: `core.selection.wrap_positions(x, box)` accepts `box=None` and replaces `deposition._wrap`.
- One Mpc: `phys.Mpc_to_cm` (3.085678e24, the same literal) replaces the `MPC_CM` constants of `core.absorbers` and
  `simulations.readers`.
- `SnapshotHeader.cosmology()`: the snapshot's flat ΛCDM as a `core.Cosmology`, used by `SimulationSkewers` and
  notebooks 05, 07 instead of the repeated expression.
- `utils.plotting`: `plot_deposition_steps` and `plot_velocity_ring` share `_skewer_cells` (labels S1.., E1.., X1..,
  colours, line centres; replaces `_ring_cells`) and `_label_tracked`; `_cell_faces` (cubes or Voronoi polyhedra) is
  shared by `plot_gas_cells` and `plot_deposition_steps`; `add_panel_letters(labels=, box=)` draws row/column letters.
- Notebooks 00, 00 (step by step) and 04 keep only short calls; their inline figure code moved into the package.
- Paths outside the repository come from `skewerjax.utils.paths` (`DATA_DIR` = `$SKEWERJAX_DATA`,
  `BENCHMARK_RAW_DIR` = `$SKEWERJAX_BENCHMARK_RAW`, default `benchmark_results/raw`, `BENCHMARK_RECORDS_DIR`):
  notebooks 00 (step by step), 05, 07, 09, the validation and benchmark scripts and the tests.
- The test runner without pytest and the markdown-math checker moved to `scripts/dev/`.

### Added
- `utils.plotting`: `plot_toy_box` (00), `plot_slab_and_skewer`, `plot_cell_crossings`, `plot_voigt_of_piece`,
  `plot_tau_stretches`, `plot_wrap_tiles`, `plot_overflux_p1d_xi` (00, step by step), `plot_grf_draws` (04),
  `shade_outside_window`.
- Tests: `test_regression.py` (+ `tests/data/regression_toy.npz`: the route and every optional mode on toy inputs),
  `test_streamed_selection.py` (the streamed selections on a synthetic HDF5 snapshot against brute force),
  `test_voigt_harris.py` (moved from the agent workspace), smoke tests of the new figure functions.

### Performance (same results)
- `select_cells_near_paths`: a coarse periodic occupancy grid of the paths skips the k-d tree for the cells that cannot
  reach them (conservative: identical selections); exact per-bin bounds; threads sized to each query.
- `select_cells_in_region` (`snapshot_cells`): the region test runs axis by axis on the surviving cells.
- `core.select_cells`: 5 radius-bin edges (instead of 33) below 10⁶ cells and threads sized to each query (thread start-up dominated).
- `_voronoi_faces`: the neighbourhood is built with array operations.

## Unreleased — notebook 07: the wrap on a ring, Si III visible (2026-10-26, session 20261026_0040)

### Added
- `plot_velocity_ring(parts, regions, v, tau_nowrap, tau_wrap, *, period, track=, cut_fraction=, zoom_kms=, tau_clip=)`:
  the periodic wrap of `velocity_period=True` on its natural geometry, θ = 2πv/P clockwise from 12 o'clock: (a) the
  no-wrap axis as an Archimedean spiral (one turn per P; below 0 and beyond P shaded), (b) the spiral collapsed onto a
  ring, with the seam and the cut ray v_c + P/2 of the cell that straddles it, (c) both zoomed on the seam (± 150
  km/s; arrows: v − P lies on the same ray, one turn in), (d) the whole skewer's τ on the spiral (no wrap) and on an
  outer ring (wrap). Same labels, colours and default tracked cells as `plot_deposition_steps` (`_ring_cells`).
  Notebook 07 shows it right after `plot_deposition_steps`, as an alternative to its rows (c)–(e).
- `plot_frames_axes(..., tau_scale={line id: k})`: display only, that line drawn as exp(−k τ) and labelled
  "τ × k for visibility". Notebook 07 §2 draws Si III with `SI_BOOST` = 10⁸ (its τ ≤ 9×10⁻⁹ on the box-centre skewer).

## Unreleased — notebook 07 on a real skewer (2026-10-25/26, sessions 20261025_2300, 20261025_2332 and 20261026_0016)

### Changed
- Notebook 07: the derivation of the sum over pieces is the opening theory (unnumbered); §1 runs the deposition step
  by step on one TNG50-4 skewer along x through the box centre, from face to face (one point per 8 km/s, coarse on
  purpose for the figure: the resolution check warns, and the same line sampled every km/s is used in §2; the cells of
  a slab from `snapshot_cells`, `select_cells`, `voronoi_segments`, `absorbers_from_components`, `absorber_pieces`,
  `cell_pieces`, `compute_optical_depth`), and describes how the code wraps (one operation: the nearest periodic image
  of every piece's detuning, `spectral.frequency_offset_GHz`, so a profile near an edge is split; the cut at P/2;
  conservation: the window holds 0.9616 of the column without wrap, 0.999999 with it, and the balance is printed); §2 (frames, global axis, observed wavelength) shows the flux F
  of Lyα and Si III on the same line, each line computed on its own window [0, P) (periodic) and relabelled on the
  global axis. The closed-form section (its checks are `scripts/validation/check_optical_depth_analytic.py`, case
  (ii)) and the application to simulation outputs are removed. Cross-references in 00, 00b, 05, 08 renumbered.
- `plot_deposition_steps(parts, bundle, regions, v_axes, tau_nowrap, tau_wrap, *, period, ...)`: rebuilt for a
  simulation skewer from the parts of `cell_pieces(..., return_parts=True)`, two columns (the start and the end of the
  path) and one step per row, every cell with one colour (its density) and one label: (a) the cells in 3D with
  velocity arrows, (b) the pieces (cell crossings and dual-segment ends), (c) every piece's dτ with its v_H → v_c
  arrow, (d) one profile on the circle of `velocity_period=True`: tracked cells (`track=`; by default the stretch
  cell that leaves the window most and the cell of the whole skewer whose profile straddles an edge most, labelled
  X1) split at the edge, the part beyond it (faded) evaluated P away in the other column (solid, arrows), the whole
  nearest-image curve as a thin line, each stretch's own τ in grey, the cut at P/2 annotated (`cut_fraction=`); (e) τ
  of the whole skewer without wrap ("not on the spectrum" outside the window), with wrap (dotted outside: the periodic
  repetition), the no-wrap τ beyond the other end moved by ∓P, and Δτ split into (i) line centres outside the window
  and (ii) tails of centre-inside pieces across an edge.
- `plot_frames_axes`: panel (b) by default relabels every line's own-frame flux at v − Δv_j (each line keeps the
  window it was computed on); `F_ref` on one `v_ref` grid is still accepted.
- `_draw_cells_3d`: the 3D cell drawing of `plot_gas_cells`, shared with `plot_deposition_steps` (no change to
  `plot_gas_cells`).

### Fixed
- Notebook 07, §2: Lyα absorbed below v = 0 on the global axis, because both lines were evaluated with
  `line_frame="reference"` on one grid holding both lines, which evaluates Lyα beyond its window (the far wings of the
  absorbers near the start). Each line is now computed on its own window and relabelled (no package change:
  `line_frame="reference"` evaluates on the grid it is given).

### Removed
- `cell_colors`, `voronoi_polygons_2d` and the toy-plane version of `plot_deposition_steps` (no other user).

### Tests
- `test_plotting::test_deposition_steps_on_voronoi_cells`: the new figure on random Voronoi cells (the end wraps to the
  start), and `plot_frames_axes` keeping the reference line within its window and moving Si III by −Δv.

## Unreleased — notebook 06: two figures simplified (2026-10-25, session 20261025_2155)

### Changed
- `plot_voigt_limits(x, sigma, gamma, curves, ref_key, *, linthresh, figsize)`: profiles (a) and residuals (b) only; the
  `expansion=` option and its panel (c), the error of the Harris expansion alone, are removed (the numbers stay printed
  in notebook 06; `voigt="harris"` and its curve in (a)/(b) are unchanged).
- Notebook 06 §3: the ratio panel against v − v_c is replaced by a residual panel under (a), sharing its axis:
  φ_linear/φ_exact − 1 with the zero line and the line centres; text of §1 and §3 updated.

## Unreleased — checkpoint up to notebook 05: the DAG matches the code, systematic tests, clean-up (2026-10-25, session 20261025_2105)

### Changed
- `plot_dag`: line profiles sit between the deposition and the optical depth (deposition → line profiles: T, v∥, a of
  the pieces; deposition → optical depth: n ℓ / a² of the pieces); new edge lines → optical depth (f and λ in the
  prefactor and the detuning, the offsets Δv for the reference axis); the edge lines → overflux is removed (the overflux
  reads no line data: the lines are already on one axis). Every edge starts and ends on the outline of its boxes
  (`_DAG_PAD`); `DAG_EDGES` lists the code dependency of each edge. Figure size (26, 6.6).
- Notebooks: the lines "Requires" / "Independent of" removed from every introduction.

### Removed (unused)
- `core.select_cells(voronoi_factor=, voronoi_min=)`, `selection_radius(voronoi_factor, voronoi_min)` (the factor is
  the constant `VORONOI_FACTOR = 2.5`); `simulations.select_cells_near_paths(voronoi_factor=, voronoi_min=,
  datasets=, progress=)` and its in-memory branch (cells in memory: `core.select_cells`), and its re-exports of
  `select_cells`, `jztree_available`.
- `select_cells_in_region(datasets=, progress=)`, `gas_to_cells(cloudy=)`; `ArepoHDF5Reader(part_type=)` (the gas
  group is `GAS_GROUP`), `read_rows(rows=)`, the duplicate `readers.GFM_METALS_ORDER`; `ion_properties(X_mu=,
  gamma_HI=, f_bar=)`; `GasCells.nbytes`.
- `grid_skewer_points(half_shift=)` (always the cell centres), `get_mass_amu(return_dict=)`; `core.cells._wrap` (now
  `core.selection.wrap_positions`).

### Tests
- New: `test_constants_and_lines.py`, `test_skewers.py` (curved paths across the periodic faces, a(x) against
  Einstein-de Sitter for every IC convention), `test_selection.py` (replaces `test_selection_backend.py`: radii,
  completeness on Voronoi and grid cells, periodic images, back ends), `test_gas_physics.py` (synthetic cells: T, n,
  H I models, CLOUDY, `gas_to_cells`), `test_snapshot.py` (TNG50-4, skipped without the data: reader, the streamed
  selection = `core.select_cells` on a region, `to_cells` = `gas_to_cells`).
- `test_toy_fields.py`: the GRF grid modes and checks, unbiased draws, `interpolate_periodic`; the `grf_cells` check of
  `test_cells.py` merged into it. `test_cells.py`: the builders and the container API. `test_plotting.py`: the DAG
  edges end on the box outlines.

## Unreleased — toy cells on the grid only, a 3D figure of the cells along a skewer (2026-10-25, session 20261025_1854)

### Changed
- `plot_gas_cells(cells, selected, path, species=None, *, view=, zoom=, decades=, alpha=)`: the cells of `selected`
  (`core.select_cells(cells, bundle.spatial_points)`) that one skewer passes through, in 3D: cubes for grid cells,
  Voronoi polyhedra (scipy, periodic, computed for the drawing only) otherwise; one row per species, columns density,
  temperature, peculiar velocity (colour v_par along the skewer, one 3D arrow per cell); edges, generating points and
  the skewer drawn in every panel; T and v colour scales shared by the rows. Replaces the plane view and the histogram
  column. `add_panel_letters` accepts 3D axes.
- `grf_cells(delta, grf, n_bar, *, temperature, velocity, turbulence, collisional)`: the cells of the grid only (its
  own Voronoi tessellation, cubes centred on the nodes).
- Notebook 04: §2 builds the grid cells and ends with the new figure; the statistics of the draws (P(|k|), ξ(|r|),
  the (k∥, k⊥) maps, the μ wedges) moved unchanged to notebook 08, §4. Notebook 05 §3 ends with the same figure on
  the TNG Voronoi cells.

### Removed
- `uniform_generators`, `density_weighted_generators`, and the `generators=`, `volumes=`, `samples_per_cell=`, `seed=`
  options of `grf_cells` (`interpolate_periodic` stays, a general utility used by `scripts/validation/check_jax.py`);
  the comparison of the three sets of cells in notebook 04.

## Unreleased — notebooks 04 and 05: one general GRF model, one figure of the gas cells (2026-10-25, session 20261025_1506)

### Added
- `plot_gas_cells(cells, species=...)`: the standard figure of a `core.Cells`, one row per species: on a plane, the
  density with the cell boundaries and the generating points, the temperature, the peculiar velocity (colour across
  the plane, arrows in it), and the density histogram of every cell. Notebooks 04 (toy cells) and 05 (snapshot) end
  with it, followed by a cell that reads the container.

### Changed
- `plot_isotropic_recovery(..., n_lin=)`: the points are compared with the input averaged over the modes of their own
  bin, at the same k, in both the P(k) panel and the ratio panel (the curve at the bin's k missed points by several
  error bars where the input bends or oscillates within a bin); optional one-mode bins at low k; ratio axis fitted
  to the data.
- `plot_kpar_kperp_maps(..., n_lin=, n_log=)`: input contours from the input binned like the measurement; empty
  rows or columns of bins merged (`_occupied_edges`). `plot_pk_mu_wedges`: the input of each wedge drawn at the same
  offset k as its points.
- `plot_nH_T_fractions`: the EOS fit drawn only over the densities of 99.5 % of the star-forming cells; labels in
  mathtext.
- Notebook 04 §1: the simple and the anisotropic GRF sections merged into one general model (theory built term by
  term), with more draws (1D 4000, 2D 1000, 3D 128³ × 96) and finer k bins; sizes in the setup cell.

## Unreleased — a fast Voigt mode, notebooks 06 and 07 restructured (2026-10-25, session 20261025_1357)

### Added
- **`voigt="harris"`** (`core.voigt.voigt_harris`, `harris_H`, `HARRIS_A_MAX`): H(a, u) to first order in the damping
  parameter, e^(−u²) + a H₁(u) with H₁ = 2u Im w(u) − 2/√π tabulated (1024 intervals on [0, 6]) and a five-term
  asymptotic series beyond u = 6, for a ≤ 10⁻³; the exact (ExoJAX) evaluation above. The scheme of fake_spectra
  (`fastvoigt.h`; Harris 1948). Max relative error 1.3 × 10⁻⁵ (a = 10⁻³); about 2× faster (1–3× across runs) than the default on a H I
  τ, slower when a bundle mixes in metal lines (the whole batch then takes the exact path). Through
  `voigt_line_profiles`, `line_profiles`, `compute_optical_depth(..., voigt="harris")`. Default unchanged.
- `plot_voigt_limits(..., expansion=)`: panel (c), the error of an approximation of H(a, u) over its range of a.
- `plot_deposition_steps(..., tau_pieces_periodic=, period=)`: panel (d), the periodic spectrum on the axis of (c).

### Changed
- Notebook 06: the theory of the line profile of a gas element (V, σ, γ, v†, λ′, v_c, ν), moved from 07; the
  "harris" mode in §1; new §3, exact against linearised detuning.
- Notebook 07: §1 the integral → the sum over pieces (theory only); §2 the deposition step by step with the periodic
  boundaries (the old §4 and its five-piece demo removed); §3–§5 renumbered. Cross-references in 00, 00b, 05, 06, 08.

## Unreleased — the field sources in `simulations`, a line-profile module, an anisotropic / BAO GRF, notebooks 06–09 (2026-10-25, session 20261025_0340)

### Added
- **`PowerLawGRF` optional terms** (all off by default; the default draws, power and variance are bit-identical):
  P(k, μ) = [P_pl(k) + P_BAO(k)] (1 + β μ²)² D(k μ σ_FoG), μ = |k∥|/|k| along `los_axis` (default: the last axis; μ = 1
  in 1D). `beta` (Kaiser-like), `sigma_fog` and `fog` (`"lorentzian"` 1/(1 + x²), `"gaussian"` e^(−x²)), and a BAO
  bump in ξ, `A_bao`, `r_bao`, `sigma_bao`: ξ_BAO(r) = A[g(r − r₀) + g(r + r₀)], added to P through its exact transform
  (closed forms in 1D and 3D, a Hankel transform by Gauss–Legendre quadrature in 2D, `bessel_j0`). P_pl + P_BAO ≥ 0 is
  checked at construction (`ValueError` naming the k). New methods: `power(k, mu)`, `power_grid`, `power_at(*k)`,
  `power_monopole(k)`, `mu_grid`, `bao_power`; the terms switched on are pytree leaves.
- **Generating points of the toy cells:** `uniform_generators(grf, M, seed=)` and `density_weighted_generators(delta,
  grf, M, seed=)` (point density (M / L³)(1 + δ_j) in the grid cell of node j, drawn exactly).
- **The snapshot as cells without skewers** (new `simulations/snapshot_cells.py`): `select_cells_in_region(reader, lo,
  hi, margin=)` (streamed, periodic sub-box), `gas_to_cells(gas, species, ext)` and `snapshot_cells(reader, species,
  lo, hi, margin=, ext=)` → `core.Cells`, with the values of `SimulationSkewers.to_cells` for the same cells
  (`check_tng_fields.py`: bit-identical, complete, exact nearest generators inside the region).
- `utils.plotting.input_power_modes`; `power_map_kpar_kperp(..., los_axis)` and `power_kmu(..., los_axis)`.
- Tests: `test_toy_fields.py` (generators; the optional GRF terms against quadrature, positivity, `power_at`,
  `power_monopole`, pytree).

### Changed (breaking)
- Moved, no aliases: `skewerjax.core.toy_fields` → `skewerjax.simulations.grf` (`PowerLawGRF`) and
  `skewerjax.simulations.toy_cells` (`grf_cells`, `interpolate_periodic`); `skewerjax.core` no longer exports them
  (import from `skewerjax.simulations`). The line widths and profiles (`voigt_widths_GHz`, `voigt_width_kms`,
  `voigt_line_profiles`, `line_profiles`) moved from `core.optical_depth` to the new `core.line_profiles` (still
  exported by `skewerjax.core`; the package attribute `skewerjax.core.line_profiles` is the function).
- The theory curves of the GRF figures (`plot_isotropic_recovery`, `plot_kpar_kperp_maps`, `plot_pk_mu_wedges`,
  `correlation_binned`, `grf_p1d_along_axis`, `tutorial.p1d_expected`, `p3d_expected_on_skewer_grid`,
  `binned_input_power`) evaluate the input at every mode with its μ (`power_at`); for an isotropic input they are
  unchanged (up to the order of the sums in |k|).
- `plot_dag`: new nodes "line profiles" and "deposition" (lines → line profiles → optical depth; gas cells + skewers
  → deposition → optical depth); no skewers → simulation edge; the toy node is "toy fields"; figure size (24, 6.6).
- Notebooks: `06_line_profiles_and_optical_depth` split into `06_voigt_and_line_profiles` and
  `07_deposition_and_optical_depth`; `07_summary_statistics` → `08_summary_statistics`; `08_fake_spectra_comparison`
  → `09_fake_spectra_comparison`. 04 adds the anisotropic + BAO configuration and rewrites its §2 (grid, uniform and
  density-weighted generating points); 05 §3 uses `snapshot_cells` (no skewers).

### Fixed
- `scripts/validation/check_jax.py` imported the module `core.line_profiles` as a package attribute, which is the
  function of the same name: it now imports the names from the module.

### Numerical changes
- None for the moves (9 baseline configurations bit-identical: GRF draws, cells, profiles, toy τ, TNG50-4 τ); none for
  `PowerLawGRF` with its optional terms off.

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
