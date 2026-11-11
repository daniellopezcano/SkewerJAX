# Performance: how SkewerJAX computes many spectra fast

This document explains the design that makes SkewerJAX faster than fake_spectra at every number of skewers on
TNG50-2 (snapshot 25, 1 km/s pixels, Lyα and Si III 1206, 32 CPU threads, float32): what each part of the code does,
why it is fast, what it costs in accuracy, and which options keep the alternatives. It is written for a physicist who
wants to know what happens to the gas between the snapshot and τ, not for a software engineer. The measurements are
those of notebook 14 (`benchmarks/results/scaling/`) and of the development runs listed in each section.

**In short** (TNG50-2, cold page cache, 1 km/s pixels, Lyα + Si III, float32, 32 threads; §6 has every entry):

| skewers | fake_spectra | SkewerJAX default, CPU | default, GPU | matched, CPU | SkewerJAX at `01d9900` (default CPU) |
|---:|---:|---:|---:|---:|---:|
| 16 | 308 s | 25.7 s | 25.5 s | 26.1 s | 28.5 s |
| 1,024 | 321 s | 58.8 s | 56.1 s | 61.6 s | 328 s |
| 16,384 (128²) | 498 s | 134.6 s | 105.1 s | 180.1 s | ~3,000 s |
| 65,536 (256²) | 1,069 s | 385.5 s | 273.0 s | 562.0 s | — |


## 1. Cost models

A code that computes N spectra from a snapshot of N_c gas cells, with N_v pixels each, pays

    T = T_fixed + N × t_skewer,   T_fixed ≈ t_read N_c + t_physics N_c',   t_skewer ≈ t_pieces + t_τ,

where N_c' is the number of cells whose gas physics is computed.

**fake_spectra** reads the whole snapshot (every needed dataset of every gas cell, ~120 GB for TNG50-2), computes the
gas physics of every cell (ionisation, temperature: 124–138 s for 1.2e9 cells), and then, per skewer, finds the
particles near the line (its index table) and adds every particle's profile on the pixels around its line centre
(its C engine: 11.6 ms per skewer on 32 threads). Its fixed cost, cold, is ~310 s at any N (I/O 166 s, physics
126 s; ~210 s when part of the snapshot is still in the page cache); its marginal cost becomes visible above ~4,000
skewers (498 s at 128², 1,069 s at 256², cold).

**SkewerJAX before this work** (commit `01d9900`) had almost no fixed cost (it read the snapshot near the skewers
only) but a large cost per skewer: the pieces of every cell were cut every 0.25 km/s (~17,000 pieces per TNG50-2
skewer, 36 ms to build with k-NN queries at every sampling point), and every piece was evaluated on every pixel (the
dense sum: 17,000 × 3,961 × 2 profile evaluations, 125 ms per skewer on the CPU). Batches of 1,024 skewers re-read the
snapshot (17 passes at 128²). It was faster than fake_spectra below ~1,000 skewers (CPU) and ~2,700 (GPU), and 5.7×
slower at 128².

**SkewerJAX now** (this document) has a fixed cost set by one read of the snapshot and the gas physics of the cells
near the skewers, and a cost per skewer that does not depend on the sampling and grows with the number of cells the
skewer crosses (~1,230 on TNG50-2), not with N_pieces × N_v.

| TNG50-2, cold, 1 km/s | fixed cost | cost per skewer |
|---|---|---|
| fake_spectra | 308 s (I/O 166 s, physics 126 s) | 11.6 ms (τ) |
| SkewerJAX, commit `01d9900`, default CPU | 28 s (16 skewers) | ~300 ms (1,024 skewers: 328 s cold) |
| SkewerJAX now (the pass) | 27 s (16 skewers) → 55 s (1,024) → 81 s (16,384) → 161 s (65,536) | — |
| SkewerJAX now, default CPU | the pass | 3.7 ms (τ; the host's crossings and contributions run behind it) |
| SkewerJAX now, default GPU | the pass | 1.9 ms (τ, mostly waiting for the host) |
| SkewerJAX now, matched CPU | the pass (and the second physics configuration, +3 s at 256²) | 6.3 ms (τ) |

The pass is not a constant: it reads every page of TNG50-2 from about 256 skewers on (~50 s cold), and beyond that it
grows with the cells within reach (23% of them at 128², 58% at 256²), whose physics it computes and which it copies into
the store.

## 2. The structural changes

### D2. Pieces are cell crossings, and each crossing absorbs with its profile integrated over it

**What it computes.** Inside one Voronoi cell the density n, the temperature T and the peculiar velocity are constant.
Along a straight path x(s) = x₀ + s t̂ the cells crossed and their boundaries are exact: the squared distance to a
generator g is (s − u_g)² + r_g², with u_g = t̂·(g − x₀) and r_g its distance from the line, so the nearest generator at
every s is the lowest of the straight lines −2u_g s + u_g² + r_g², and their lower envelope gives every crossing,
s* = (u_i + u_j)/2 + (r_i² − r_j²)/(2(u_i − u_j)) (`core.crossings`). A crossing [s_lo, s_hi] is one piece.

Across a crossing only the line centre moves, with the Hubble flow, v_c(s) = V_H(s) + v_pec, from v_lo = v_c(s_lo) to
v_hi = v_c(s_hi). The optical depth of the crossing is the integral of the profile over those line centres,

    τ_cr(v) = 𝒞 n ∫ a(s)⁻² φ(ν(v − v_c(s))) ds
            = A ∫ w(t) φ(ν(v − v_c,0 − t)) dt / W,    A = 𝒞 n ∫ a⁻² ds,   W = |v_hi − v_lo|,

with t the offset of the line centre from the middle of the span v_c,0 and w(t) the weight per unit line centre,
a(s)⁻² ds/dv_c, normalised to 1. In Doppler units u = ν / (√2 σ) the profile is the Voigt function, φ = H(a, u) /
(√(2π) σ), and the mean of H over an interval has a closed form in Harris' expansion H = Σ aⁿ H_n(u)
(`core.voigt_integral`). With w(u) = e^(−u²) + (2i/√π) F(u) on the real axis (F Dawson's function), H_n = Re[iⁿ w⁽ⁿ⁾/n!]
integrates to the imaginary part of the previous term:

    Φ(u) = ∫₀ᵘ H du = (√π/2) erf u − a (2/√π) F(u) + a² u e^(−u²) − a³ (2/(3√π)) [(1 − 2u²) F(u) + u]
                      + a⁴ (u/2 − u³/3) e^(−u²) + O(a⁵),

so that ⟨H⟩ over [u_lo, u_hi] = [Φ(u_hi) − Φ(u_lo)] / (u_hi − u_lo). The first order (a ≤ 10⁻³: every Lyα crossing)
keeps two terms, the fourth order (a ≤ 0.05: cold Si III) all five, as the point profile does (`core.voigt`). Where
a > 0.05 (very cold metal gas: rare) the crossing is integrated numerically, 3-point Gauss–Legendre on sub-pieces of
half a Doppler width with the exact (ExoJAX) function.

Three effects of first order in W/c and W V″/V′² make the weight w(t) non-uniform, and the integral keeps them:
- the comoving-to-physical factor a(s)⁻² changes along the cell: d ln a⁻²/dv_c = 2/c;
- the Hubble flow is slightly curved, so the line centres are uniform in s, not in v_c: d ln(ds/dv_c)/dv_c = −V″/V′²;
- the exact detuning ν = (c/λ)(e^(−Δ/c) − 1) maps the line centres to u with the Jacobian e^(−Δ/c), and the centre of
  the interval in u lies above u(v_c,0) by h W/(4c).

The first two are the *tilt* g of the contribution (`LineContributions.tilt`, `Pieces.tilt_a2`, `Pieces.tilt_flow`),
taken from the frame at the two ends and the middle of the crossing; the exact detuning adds −1/c. The integral is then

    τ_cr(v) = A / (√(2π) σ) [⟨H⟩ + g (dt/du) ⟨(u − ū) H⟩],

the first moment ⟨(u − ū)H⟩ of the Gaussian term in closed form, [(e^(−u_lo²) − e^(−u_hi²))/2 − ū (√π/2)(erf u_hi −
erf u_lo)] / (u_hi − u_lo). Without these terms Lyα differs from the converged result by rms ΔF = 2.3e-6 on TNG50-4
(τ ~ 1 makes Lyα sensitive to a relative error of 10⁻⁵); with them, by 2.4e-9.

**Numerics.** The difference of Φ loses precision for a narrow interval (Φ ~ 1, the difference ~ H Δu) and in the far
tails (erf → 1). So intervals of half width h ≥ 0.2 take the difference with the erf part written through erfc(|u|) on
each side of 0 (no cancellation), and narrower ones the Taylor series of the mean about the midpoint, ⟨H⟩ = H + (h²/6)H″ +
(h⁴/120)H⁗ + (h⁶/5040)H⁽⁶⁾, with the derivatives in closed form (Hermite polynomials of the Gaussian, Chebyshev series of
the Dawson terms). The Chebyshev degrees follow from the accuracy each term needs. Against numerical quadrature of
`scipy.special.wofz` the mean is exact to the accuracy of Harris' expansion itself (≤ 2e-7 of the peak for a ≤ 0.05,
1e-6 at a = 10⁻³ where the first order stops) (`tests/core/test_tau_engine.py`).

**Why it is faster.** A TNG50-2 skewer crosses ~1,230 cells; cutting them every 0.25 km/s made ~17,000 pieces. The
crossings need no sampling points, no k-NN queries and no selection radius in ds: the candidates of every skewer are the
cells whose generator lies within the reach 2.5 V^(1/3) of the line (the rule of every selection of the package; the
owners of the crossings lie within r/h ≤ 1.39 on TNG50-4), found by index arithmetic (§D1). The envelope over ~14,000
candidates per skewer takes ~1 ms.

**Accuracy.** The crossings are identical to the old exact envelope pieces joined per cell (64 TNG50-4 skewers: the
same cells, the same boundaries to 7e-15 cMpc). The integrated profile removes the discretisation error of the
midpoint sum, O(W²/12b²) per piece: against the converged result (the float64 midpoint sums at 0.125 and 0.0625 km/s,
Richardson-extrapolated) the new default agrees to rms ΔF 2e-9 in float64 and 4e-7 in float32 (Lyα; Si III 6e-8), while
the old default at its finest sampling, 0.125 km/s, differed from it by rms 7.7e-7 / max 9.6e-6 (Lyα; Si III max 2.7e-5,
its discretisation) in float64 and by rms 1.9e-6 / max 1.6e-5 from its own float64 in float32 (its rounding).

| against the converged result (64 TNG50-4 skewers) | Lyα rms / max ΔF | Si III rms / max ΔF |
|---|---|---|
| new default, float64 | 2.4e-9 / 1.8e-7 (both lines) | |
| new default, float32 (the package's precision) | 4.2e-7 / 4.4e-6 | 5.9e-8 / 7.1e-6 |
| old default (0.25 km/s), float32: what it computed | 3.9e-6 / 4.8e-5 | 7.3e-7 / 9.5e-5 |
| old default at 0.125 km/s, float64 (its discretisation) | 7.7e-7 / 9.6e-6 | 2.0e-7 / 2.7e-5 |
| old default at 0.125 km/s, float32 | 2.1e-6 / 2.1e-5 | 2.2e-7 / 3.6e-5 |
| old default at 0.125 km/s, float32 vs its own float64 (its rounding) | 1.9e-6 / 1.6e-5 | 1.3e-7 / 2.5e-5 |

The converged result is the float64 midpoint sum at 0.125 and 0.0625 km/s, Richardson-extrapolated. The spec's literal
reference, "the old default at 0.125 km/s", carries its own discretisation error (2.7e-5 at its worst Si III pixel) and,
in float32, its rounding (1.6e-5), so it cannot be matched to the target's 2e-5 by any code: the new default is compared
with the converged result, and so is the old default. Against the literal reference the new float32 default differs by
rms 8.6e-7 / max 1.5e-5 (Lyα) and 2.1e-7 / 2.8e-5 (Si III): the rms within the target's
1e-6, the Si III maximum above its 2e-5 by the reference's own error at that pixel.

**Options.** `profile_integration="midpoint"` (`core.compute_optical_depth`) evaluates every piece at its line centre;
`deposition="voronoi_segments"` (`core.cell_pieces`) keeps the old pieces cut at the dual segments of the sampling
points (curved paths always take them: §3). Both reproduce the old default.

### D3. A windowed engine with exact far wings

**What it computes.** The old sum evaluated every piece on every pixel. Most of those evaluations are far wings: the
Gaussian core of a contribution is negligible beyond a few Doppler widths. The engine (`core.tau_engine`) evaluates each
contribution exactly on a window of pixels around its line centre, |k − c| ≤ D, and adds the rest of its profile, the
Lorentz wing, analytically. Beyond the core the profile is Harris' asymptotic series; in the exact detuning
ν = −κ Δ′ (κ = 1/λ, Δ′ = c(1 − e^(−Δ/c))):

    A φ(ν) = w / Δ′² [1 + (3/2) b² / Δ′² + O(b⁴/Δ′⁴)],    w = A γ / (π κ²),   b = √2 σ / κ,

which is the same function of Δ for every contribution of a line up to the weights w and w b². So the far wings of all
the contributions of a skewer are one convolution: each contribution deposits four moments at its centre pixel c — w,
w δ, w (δ² + W²/12) and (3/2) w b² (δ the offset of its line centre from the pixel, W its extent) — and the deposits are
convolved, by FFT, with the kernels K, −K′, K″/2 and 1/Δ′⁴ (K = 1/Δ′²), set to zero within D_x pixels (the smallest
window of the line). Inside each window the kernel subtracts exactly what the FFT adds on its ring D_x < |k − c| ≤ D, so
the two parts never overlap: the result is the exact profile in the window and the asymptotic series beyond.

**Why truncation would not do.** Dropping the wings beyond a window misses the sum of the Lorentz wings of the whole
forest, (γ_v/π) Σ τ_int,i / Δ_i² ≈ 1.5e-3 / Δ_min [km/s]: ~1e-5 in τ everywhere for windows of ±150 km/s. That is an rms
ΔF of several 10⁻⁶, above the tolerance; the FFT costs ~1% of the engine.

**The window bound** (`wing_tolerance` ε, default 10⁻⁹, `compute_optical_depth`). At the first pixel outside the window,
at a distance Δ_e ≥ (D + 1) dv − |δ| − W/2 from every line centre of the piece, the far term omits at most a relative
3.75 (b/Δ_e)⁴ + 15 (b s/Δ_e²)² + 4 (s/Δ_e)³ + 5 (s/Δ_e)⁴ of the wing (s = |δ| + W/2) and the Gaussian core,
τ₀ e^(−(Δ_e/b)²). D is the smallest level of 4, 8, ..., 8192 pixels that brings w/Δ_e² × (relative error) + τ₀ e^(−Δ_e²/b²)
below ε; beyond it the omitted terms fall as Δ⁻⁶. A typical Lyα crossing (τ₀ ~ 1) takes D = 128–256 pixels at
1 km/s, a Si III one 4–16; a damped absorber whose window would exceed half the grid takes every pixel, with no far
term. With ε = 10⁻⁹ the windows and far wings reproduce the dense sum to 4e-8 in τ (float64,
`tests/core/test_tau_engine.py`). `return_info` reports the windows (`engine`: the levels used, D_x, the full-grid
contributions).

**Periodicity.** A `velocity_period` equal to the span of the grid (N_v dv, the periodic spectrum of a box) wraps the
window indices and makes the far convolution circular. Without a period the convolution is linear (zero-padded over half
a grid beyond each end; line centres farther off the grid take the whole grid). Any other period takes every pixel for
every contribution with the nearest image of each detuning (exact; the image boundary half a period from each line
centre does not fall on the pixel offsets of one shared kernel).

**On the CPU,** XLA's scatter-add is serial (5–7 ns per update, and it absorbs the evaluation it consumes), so the
kernels return their windows and the host adds them with a numba loop parallel over skewers (each thread owns its
skewers' rows). On a GPU the device scatter-adds (atomics). Small problems (N_contributions × N_v ≤ 2e7) take every
pixel: the same exact result with fewer kernels to compile.

**Unified with the per-cell schemes.** fake_spectra's early stop (`tautail`) and pixel average (`pixel_average`) are the
group mode of the same engine (`tau_engine.per_cell_line_tau`): the pieces of a cell form one absorber on a window
escalated until fake_spectra's walk stops inside it. `core.per_cell` is gone.

### D1. One read of the snapshot: the streaming route

**What it does.** The old route selected the cells near each batch of 1,024 skewers with a k-d tree over the
sampling points, after reading the positions of every cell, and read the other datasets of the cells it kept: each
batch re-read the snapshot (17 passes at 128² skewers). The streaming route (`simulations.stream`) reads the snapshot
once, in file order, in blocks of 2 million rows on parallel threads. For each block it

1. reads the geometry (positions, masses, densities: 32 bytes per cell) whole, and finds the cells within reach
   R = 2.5 V^(1/3) of some skewer by index arithmetic: the skewers of an axis are points of the plane perpendicular to
   it, bucketed on a grid of the plane, and a cell at (x, z) looks only at the buckets within ±R (`core.crossings.
   line_pairs`, numba; no tree, no sampling points);
2. reads the other datasets (velocities, internal energy, electron and H I abundances, metals, star formation: 68 bytes
   per cell) for those rows only: one contiguous read where they touch more than half the pages of the block, an indexed
   gather through a memory map otherwise;
3. computes the gas physics of those cells only, once per physics configuration (the default and the matched one share
   the pass), with fused loops (`simulations.host_physics`: the formulas of `gas_physics`, the same float32 values as
   `gas_to_cells`, numba without the GIL; the CLOUDY interpolation stays SciPy's), and keeps a compact record per needed cell
   (position, size, peculiar velocity, id: 48 bytes; n and T of each species: 16 bytes per configuration for Lyα +
   Si III) with the tiles that need it (8 bytes per pair).

The skewers are then processed in **tiles** of about 1,024 neighbouring skewers (squares of the skewers' plane): the
store hands each tile its cells (a counting sort of the (cell, tile) pairs of the pass), and the host builds the
crossings and contributions of tile k + 1 (three threads) while the engine sums tile k.

**Why it is faster.** The pass costs one read of the snapshot. TNG50-2 holds 1.2e9 cells (~120 GB of the datasets
needed); the NVMe disk reads ~2.5 GB/s cold, so ~50 s is the floor of any code that reads it, and fake_spectra pays it
too (plus its physics on every cell). From 32² skewers up the cells within reach are spread over nearly every page of
every file, so the pass reads nearly everything and its cost no longer grows with N; below that, the rest is gathered
row by row and the pass reads less. The gas physics runs on the needed cells only: 23% of TNG50-2 at 128² skewers. Written as NumPy expressions it
writes a temporary array per operation, and many threads converting blocks at once were limited by that memory traffic
and the interpreter: 2.6× on 8 threads, nothing beyond (≈100 ns per cell for both configurations, ≥80 s of the 128²
pass). The fused loops take 30 ns per cell on 8 threads (8 s at 128²), so the pass is again bound by the disk.

| TNG50-2, cold page cache | old route | streaming route |
|---|---|---|
| 32² skewers, default CPU | 328 s (2 batches) | 59–63 s (pass ~52 s) |
| 128² skewers, default CPU | 3,015 s (17 batches) | 130 s (pass 74 s, shared with the matched physics) |

**Options.** `simulations.snapshot_optical_depth(..., strategy=)`: "stream" for axis-aligned straight skewers,
"gather" (the k-d tree selection followed by `core.compute_optical_depth`) for any path, "auto" (default) takes the
first when it applies. `tile_size` sets the skewers per tile (the memory of one tile against the fixed cost per tile),
`threads` the threads of the pass.


### D4. The matched mode on the same route

**What it computes.** The matched mode reproduces fake_spectra's own calculation: every cell is a sphere of radius
h = V^(1/3) touching the line along a chord of half length z = √(h² − r²), sampled at 7 points with the top-hat
weights, in fake_spectra's frame (the linear Hubble flow, its constants, its density conversion), and each cell is one
absorber whose walk over the pixels stops at the first pixel below `tautail` = 10⁻⁷ on each side, with the trapezoid
over each pixel for cells narrower than two pixels (`pixel_average`). The mode keeps all of it; what changed is how it
is computed.

- **The chords** come from the same candidate pairs as the crossings (`core.crossings.straight_chords`, reach h: the
  spheres that touch the line), in the same pass and tiles as the default (`simulations.stream`: the matched physics is
  the second configuration of the pass), with fake_spectra's periodic image (the one nearest the middle of the path).
- **One contribution per chord**: its 7 points are equally spaced in line centre (the chord is straight and the flow
  linear), so a chord is its middle line centre, its spacing and its total amplitude, and the kernel generates the 7
  points (`LineContributions.spacing`). The host prepares 7× fewer rows.
- **The walk on a window** (`tau_engine.per_cell_line_tau`): each cell is evaluated on a window of pixels around its
  centre pixel sized from its peak and its damping wing (where τ₀ a / (√π u²) falls below `tautail`), the walk is a
  min / max reduction over the window (the first pixel below `tautail` each way), and a cell whose walk reaches the
  edge of its window is done again with the next window (4 … 32,768 pixels): the result is the full-grid walk's.
- **Core and wing**: beyond 10 Doppler widths from its extreme points every point of a cell is in its Lorentz wing, so
  there the cell is evaluated once per pixel from four moments of its points (the series of D3) instead of 7 times;
  the core, 32–256 pixels, is evaluated point by point. With 8 Doppler widths instead of 10 the agreement with
  fake_spectra degrades (Lyα rms 7.7e-7), with 10 it is unchanged.
- **On the CPU** the windows are added on the host in parallel over skewers (as in D3).

| matched, CPU | τ per TNG50-2 skewer |
|---|---|
| old (dense per-cell walk, 7 pieces per chord) | 126 ms |
| windows, reductions, compact chords, core/wing, one CPU device | 8.8 ms |
| the same on two CPU devices (§5) | 6.3 ms |

**Accuracy.** Matched against fake_spectra on 64 TNG50-4 skewers: Lyα rms ΔF 1.3e-7 (old 2.5e-7), max 3.9e-6; Si III
rms 5e-9. Over every skewer of the benchmark: Lyα rms ΔF 6.7e-8 (16 skewers) to 1.9e-7 (16,384 and 65,536 TNG50-2
skewers; 2.0e-7 on TNG50-4 and -3), Si III 6e-9 to 3.3e-8, within the 3e-7 of the target. The largest single-pixel
difference is 1.3e-4 (Lyα, 65,536 skewers).


### D5. Fixed costs and few skewers

With few skewers the cost is the pass, which reads the geometry of every cell (32 bytes per cell: 38 GB for TNG50-2)
to find the cells within reach, and then the other datasets of a few rows only (gathered page by page).

| TNG50-2, default CPU, cold | old route | stream | gather (k-d tree) |
|---|---|---|---|
| 4² = 16 skewers | 28.5 s | 26.5 s | 27.5 s |
| 16² = 256 skewers | — | 62.7 s | 65.3 s |
| 32² = 1,024 skewers | 328 s | 59–63 s | — |

Three things keep the fixed cost low.

- **One pass**, as for many skewers (§D1). From about 16² skewers on, the rows needed touch most pages, and the pass reads
  nearly everything: 60 s cold is then the price of reading TNG50-2 once, whatever N.
- **Few compilations.** Every kernel has fixed shapes: the rows of a tile are padded to a power of two, the chunks of
  contributions have fixed sizes, and the far-wing domain is fixed per grid. The kernels compile once per process and
  window level, about 6 s for the default entry and 31 s for all three entries on both devices (the benchmark reports
  this as `jit_s`, apart from the run). Problems below 2e7 contributions × pixels take every pixel for every
  contribution: the same exact result from far fewer kernels.
- **No persistent caches.** Nothing is kept on disk between runs: no compiled kernels (numba without `cache=True`, no JAX
  compilation cache) and no snapshot data. A cold run is a first run.

`strategy="auto"` takes the streaming route for axis-aligned skewers at any N: it is never slower than the gather route
in these measurements.


## 3. The routes

Every route ends in the same engine (`core.tau_engine`); they differ in how the gas reaches it.

| source | pieces | engine | when |
|---|---|---|---|
| snapshot, axis-aligned skewers (`simulations.snapshot_optical_depth`, `strategy="auto"` → "stream") | crossings (default) or chords (matched), per tile | windowed (default), per-cell (`tautail`, `pixel_average`) | the benchmark; any grid of box-crossing skewers |
| snapshot, any other paths (`strategy="gather"`) | k-d tree selection, then as below | as below | straight paths in any direction, curved paths, light cones |
| `SkewerBundle` + `Cells`, straight paths (`core.compute_optical_depth`) | crossings (`core.crossings`) | windowed | notebooks, toy boxes, small samples |
| `SkewerBundle` + `Cells`, curved paths | the dual-segment pieces of the sampling points (`deposition="voronoi"` falls back to them; the crossings of a curved path are not straight-line envelopes) | windowed, integrated profile over each piece | light cones, curved paths |
| `Pieces` (`core.cell_pieces`, any deposition) | as given | windowed (uniform grid, concrete arrays) | custom depositions |
| under `jax.jit`, `per_piece=True`, non-uniform grid | as given | dense sum (every piece on every pixel, integrated profile) | differentiation, diagnostics |

The streaming route takes straight skewers along one axis of the box (the reach test is index arithmetic on the plane
perpendicular to it); `choose_strategy` sends everything else to the gather route, whose selection is the k-d tree of
the sampling points (`select_cells_near_paths`), unchanged. Curved skewers keep the general path: the sampled
dual-segment pieces (`voronoi_segments`), now summed by the windowed engine with the integrated profile over each
piece, so their accuracy no longer depends on the sampling as much as before (§D2) and their cost no longer scales with
N_pieces × N_v (§D3).


## 4. Memory

The memory of a run is the compact store of the cells within reach, the blocks being read, and the spectra.

| TNG50-2, both physics configurations | 128² skewers | 256² skewers |
|---|---|---|
| cells within reach | 23% (2.7e8) | 58% (6.9e8) |
| compact store | 24 GB | 61 GB |
| peak during the pass | 28 GB | 74 GB |
| peak of the run (all three entries, benchmark process) | 45 GB | 83 GB |

- **The store** holds, per needed cell, its position (float64), size, velocity and id (48 bytes), and for each physics
  configuration n and T of each species (16 bytes for two species), plus 8 bytes per (cell, tile) pair (12 while the store is built). A run with one
  configuration needs 16 bytes per cell less.
- **The allocator.** The pass allocates and frees many arrays of a few MB on 32 threads. glibc's default raises its mmap
  threshold to the largest freed block, after which such arrays come from per-thread heaps that keep their pages. At
  128² that kept 19 GB beyond the store (peak 48 GB). The pass fixes the threshold at 1 MB
  (`simulations.stream.release_memory`) and trims the heaps after assembling the store: peak 28 GB, the pass no slower.
  The tiles then allocate many short-lived arrays of 1–30 MB per chunk of contributions, where a fresh mapping for each
  costs page faults (the matched engine ran 1.9× slower at 1 MB): when the pass ends the threshold goes to 32 MB,
  glibc's own steady state.
- **The tiles** add one tile's crossings and contributions at a time (three tiles in preparation) and the spectra,
  N × N_v × 2 lines × 4 bytes per entry (2.1 GB at 256²).
- **The GPU** holds one tile's contributions and spectra: 1.3 GB at its peak at 128², 2.4 GB at 256².

The old route held the selected cells of a batch of 1,024 skewers with every field and their sampled pieces: 125 GB of
resident memory at 32² in the matched mode.


## 5. CPU and GPU

The host does the same work for every device: the pass, the crossings or chords, the contributions. Only the sum on
the pixels runs on the device, and its two forms differ.

- **CPU.** XLA's scatter-add on the CPU is serial, so the kernels return each contribution's window and the host adds
  the windows into the spectra with a numba loop that is parallel over skewers. XLA runs one computation at a time per
  device, on part of its thread pool. Exposing the CPU as two devices (`skewerjax.config.configure_devices`, default 2)
  lets the engine deal its chunks to both, so two kernels overlap and the cores stay busy. Matched-mode τ of 1,024
  TNG50-2 skewers takes 8.6 s on one device and 5.9 s on two (12.6 cores busy on average with one), with no gain
  beyond two. The default engine takes the same time on one or two devices and is slower on four.
- **GPU.** The kernels scatter-add on the device (atomics), and the spectra stay there until the tile ends. The host
  prepares the next tiles while the GPU sums the current one; the GPU's τ stage still includes the engine's host side
  (§7). The
  peak device memory is 1.3 GB at 128² TNG50-2 skewers (one tile of 1,024 skewers × 3,961 pixels × 2 lines, and the
  chunks in flight) and 2.4 GB at 256² (the far-wing convolution of the larger tiles).
- **One process per GPU.** Importing SkewerJAX starts JAX (exojax runs JAX code at import), and JAX on a GPU reserves
  75% of its memory at start unless `XLA_PYTHON_CLIENT_PREALLOCATE=false` is set. A second process on the same GPU,
  even one that computes nothing, then leaves the first ~1.4 GB of a 12 GB card. The benchmark's launcher therefore
  runs JAX on the CPU (`JAX_PLATFORMS=cpu`); before it did, the 256² point ran out of GPU memory.
- **Agreement.** The two devices add in different orders. Over all 16,384 skewers at 128², GPU vs CPU differs by rms
  ΔF 4.1e-7 (Lyα) and 6e-9 (Si III), within the 2e-6 of the target; the largest single-pixel difference is 2e-4 (Lyα)
  and 3e-6 (Si III). Over all 65,536 skewers at 256²: rms 7.6e-7 and 1e-8, largest single-pixel difference 7.3e-4
  (Lyα) and 1.7e-5 (Si III).


## 6. Results

Notebook 14 shows these records (`benchmarks/results/scaling/`, `scripts/benchmarks/benchmark_scaling.py`); notebook 13
runs the four entries on one grid. Every run is cold (the snapshot dropped from the page cache before each code reads
it), on the i9-13900KF (32 threads, 125 GB) with the RTX 3060 (12 GB), float32, two CPU devices. The SkewerJAX times
exclude the JIT compilation, measured apart on TNG50-4 (`jit_s`: 17–31 s, 99 s for the five grids of C1).

**Block B, the number of skewers** (TNG50-2, 1 km/s). The targets of this work: from 16 to 1,024 skewers at least as
fast as before (≤ 15% slower anywhere) and ≤ 0.5× fake_spectra; ≤ 200 s at 128² (GPU ≤ 120 s); ≤ 700 s at 256² (GPU
≤ 400 s) with a peak memory ≤ 100 GB.

| skewers | fake_spectra | default CPU | default GPU | matched CPU | before (`01d9900`, cold): CPU / GPU / matched | target | met |
|---:|---:|---:|---:|---:|---|---|:---:|
| 16 | 308.2 s | 25.7 s | 25.5 s | 26.1 s | 28.5 / 26.8 / 27.8 s | ≤ before + 15%, ≤ 154 s | ✓ |
| 256 | 312.8 s | 45.4 s | 44.5 s | 46.6 s | 90.7 / 63.2 / 83.9 s | ≤ before + 15%, ≤ 156 s | ✓ |
| 1,024 | 321.4 s | 58.8 s | 56.1 s | 61.6 s | 328 / 160 / 352 s | ≤ before + 15%, ≤ 161 s | ✓ |
| 4,096 | 358.1 s | 69.1 s | 60.2 s | 80.7 s | — | faster than fake_spectra | ✓ |
| 16,384 | 498.1 s | 134.6 s | 105.1 s | 180.1 s | ~3,000 / 1,260 / 4,490 s (30% cached) | ≤ 200 s, GPU ≤ 120 s | ✓ |
| 65,536 | 1,069.2 s | 385.5 s | 273.0 s | 562.0 s | — | ≤ 700 s, GPU ≤ 400 s | ✓ |

Peak memory at 65,536 skewers: 83 GB (the benchmark process, fake_spectra's leftovers included; the pass alone, 74 GB).
The 16- and 256-skewer points are the faster of two runs (the I/O of a few rows is noisy); fake_spectra's two runs differ
by < 1 s.

**Block A, the resolution** (128², 1 km/s): TNG50-4: fake_spectra 30.6 s, default CPU 25.3 s, GPU 10.4 s, matched
42.4 s; TNG50-3: 87.3, 51.0, 30.7, 76.6 s; TNG50-2: the row above. fake_spectra's fixed cost grows with the number of
cells (it reads and converts them all); SkewerJAX's cost per skewer grows with the cells a skewer crosses.

**Block C1, the pixel width** (TNG50-2, 128²): 1, 2, 4, 8, 16 km/s: fake_spectra 498, 443, 435, 489, 586 s; default
CPU 134.6, 118.5, 109.7, 104.9, 102.7 s; GPU 105.1, 103.1, 102.2, 101.2, 100.2 s (the pass, 80 s, is shared; matched
on the 1 km/s grid only).

**Block C2, the midpoint option's sampling** (64 TNG50-2 skewers, float64, against the integrated profile on the cell
crossings, 9.5 ms of τ per skewer):

| spacing | pieces per skewer | τ per skewer | Lyα rms / max ΔF | Si III rms / max ΔF |
|---|---:|---:|---|---|
| 0.125 km/s | 32,900 | 30.4 ms | 8.6e-7 / 1.1e-5 | 1.9e-6 / 2.5e-4 |
| 0.25 km/s (the earlier default) | 17,100 | 16.2 ms | 3.3e-6 / 3.6e-5 | 6.2e-6 / 7.6e-4 |
| 0.5 km/s | 9,160 | 9.3 ms | 1.2e-5 / 1.2e-4 | 1.8e-5 / 2.4e-3 |
| 1 km/s | 5,200 | 6.0 ms | 4.4e-5 / 3.4e-4 | 4.7e-5 / 5.8e-3 |
| 2 km/s | 3,220 | 4.3 ms | 1.3e-4 / 1.1e-3 | 4.8e-5 / 5.7e-3 |
| 4 km/s | 2,230 | 3.8 ms | 3.5e-4 / 3.1e-3 | 6.5e-5 / 9.4e-3 |

**Agreement over every benchmark skewer.** GPU vs CPU: rms ΔF ≤ 7.6e-7 (Lyα, 256²) and ≤ 2e-8 (Si III), target 2e-6.
Matched vs fake_spectra: Lyα rms 6.7e-8 to 1.9e-7 on TNG50-2, 2.0e-7 on TNG50-4 and -3; Si III ≤ 3.3e-8; target 3e-7.
The default against the converged result: §D2.

**The whole benchmark** (blocks A, B, C1 and C2) took 2 h 7 min (18:31–20:38 on 2026-11-10), plus 35 min for the
256² point, measured again after the launcher's fix (§5).

## 7. Limits and ideas

- **TNG50-4, matched mode.** On the smallest box fake_spectra's fixed cost is 6 s, and the matched engine's τ (2.4 ms per
  skewer, 39 s at 128²) makes the matched entry slower than fake_spectra (42.4 vs 30.6 s); the default CPU entry is
  just faster (25.3 s), the GPU 3× faster. The targets of this work are for TNG50-2.
- **The pass is bound by the disk.** Cold, TNG50-2 reads at 2.5 GB/s: ~50 s for every page, which the pass reaches from
  ~256 skewers on. Below that it reads the geometry of every cell (38 GB) and the other datasets of a few rows; a
  spatially sorted snapshot, or an index of the cells by region, would read less.
- **Memory grows with the cells within reach.** The store holds 88 bytes per needed cell with two physics
  configurations (61 GB at 256², 58% of the cells): 512² would need nearly every cell, ~110 GB, above this machine's
  125 GB with the rest of the run. One configuration per pass (16 bytes per cell less), float32 positions relative to
  the tile, or tiles processed as they complete would lower it.
- **The GPU entry is bound by the host.** Its τ stage takes 1.9 ms per skewer at 256² (the CPU's 3.7 ms), and includes
  the engine's host side (sorting the contributions into windows, dispatching the chunks); meanwhile three host threads
  prepare the next tiles' crossings and contributions, 4.1 thread-ms per skewer. Computing the crossings and the
  contributions on the device would bring the GPU entry closer to the pass.
- **The matched mode** keeps fake_spectra's per-cell walk (`tautail`), which costs 6.3 ms per skewer, 1.7× the default
  engine; it exists to reproduce fake_spectra, not to be fast.
- **The accuracy target's literal reference.** Against the earlier default at 0.125 km/s in float64, the new default
  meets the rms (8.6e-7 ≤ 1e-6) but not the maximum for Si III (2.8e-5 > 2e-5): that reference is itself 2.7e-5 from the
  converged result at that pixel (§D2), so the comparison with the converged result is the meaningful one (rms 4.2e-7,
  max 4.4e-6 in float32).
