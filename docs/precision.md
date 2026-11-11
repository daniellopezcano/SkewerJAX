# Precision: float32 by default

SkewerJAX computes in **float32 by default** and in float64 on request, for validation:

```python
import skewerjax.config

skewerjax.config.set_precision("float64")  # or: export SKEWERJAX_PRECISION=float64
with skewerjax.config.precision("float64"):  # for a block only
    ...
```

`set_precision("float64")` turns JAX's `jax_enable_x64` on (float64 needs it); `"float32"` leaves the flag as it is and
casts what the package sends to the device. `skewerjax.config.real_dtype()` is the dtype of the device kernels.

## What follows the policy, and what does not

| Stage | Precision | Why |
|---|---|---|
| τ sum (`compute_optical_depth`, the per-cell kernel of `pixel_average` / `tautail`), line profiles, the arrays of the statistics | the policy (float32 default) | the cost: the Voigt profile of every piece at every pixel |
| path coordinates, a(s) of `SkewerBundle` (`Cosmology.solve_a_of_x_host`, DOP853, rtol 1e-12), the frames, the Hubble velocities V_H = c ln(a†/a) of the line centres (`Pieces.hubble_kms`) | float64 NumPy / SciPy, always | cumulative sums and differences of nearly equal numbers times c (below) |
| selection, deposition (`voronoi_segments`, `tophat_segments`, `tophat_chord_segments`), absorbers | float64 NumPy, always | host geometry |
| per-cell gas physics (`simulations.gas_physics`, `gas_to_cells`, `host_physics`) on host data | float64 NumPy / numba (JAX for JAX inputs: jit, grad) | the dynamic range of the products (below) |
| the streaming route's store (`simulations.stream`): n a³, T, velocity and size of every cell | float32, always (positions float64) | the snapshot's own precision (TNG stores these datasets in float32); half the memory of float64 |

A `SkewerBundle` built from concrete values keeps float64 host copies of its points, distances and scale factors
(`SkewerBundle.host`), which `cell_pieces` and `bundle_frame` read; its JAX arrays are in the policy dtype. One built
under `jit` (traced) solves a(x) with diffrax in the working dtype and stays differentiable.

## The audit

Each item of the float32 audit, what it found and the fix (measured on TNG50-4, four skewers along y, notebook 12's
calls, against the same calls in float64).

1. **Velocity offsets.** The line centre of a piece is v_c = V_H + c (ln(1−β) − ln(1+β))/2. Computed in float32 from the two
   scale factors, V_H = c ln(a†/a_p) would carry their ratio's ~6e-8, an error of ~0.02 km/s
   (1 % of the Doppler width of Si III at 1e4 K, 2.4 km/s). **Fix:** V_H is computed in float64 on the host
   (`absorber_pieces`; `compute_optical_depth` fills it for `Pieces` built by hand) and stored per piece (`Pieces.hubble_kms`); the kernel adds the peculiar part (log1p, exact to
   float32 at β ~ 1e-3) and the detuning uses `expm1(−Δv/c)` (Δv/c ~ 1e-5: exact to float32 relative). What remains is
   the rounding of the velocity coordinate itself (v and V_H up to ~4000 km/s: 2.4e-4 km/s, 1e-4 of b for Si III).
2. **Cumulative coordinates along long paths.** The path coordinate s is a cumulative sum over ~16 000 points; in
   float32 it errs by ~sqrt(N) eps s ~ 4e-4 cMpc (0.03 km/s of Hubble flow). **Fix:** `path_geometry` runs in float64
   NumPy for host arrays (JAX arrays keep JAX); every host caller passes NumPy.
3. **Dynamic range of the gas physics.** ρ X for pristine gas reaches ~1e-39 g cm⁻³ and the CLOUDY fractions go down to
   1e-30, so the Si III density of metal-poor cells falls into float32's denormal range (below 1.2e-38), where it loses
   every digit: up to 100 % per cell, and **11 % in τ** on a metal-poor TNG50-4 sightline (τ_SiIII ~ 1e-8). **Fix:** the
   per-cell physics dispatches on its inputs (`gas_physics._xp`): NumPy float64 for host data; the densities are also
   formed as (ρ / A m) X.
4. **Products of the τ amplitude.** C f n a⁻² l with n down to ~1e-30 cm⁻³: the column n·l (~1e-10 to 1e10) is formed
   first, then the per-line constant C f (`optical_depth_kernel`), so no intermediate leaves the float32 range.
5. **Voigt arguments and wings.** ExoJAX's `voigtone` in float32 against SciPy's `wofz`: max relative error 2.1e-6 over
   u ∈ [0, 3000], a ∈ [1e-4, 3e-2] (float64: 1e-6, ExoJAX's own); Harris' expansion 6e-6 (1.4e-5 at a = 1e-3), the same
   in both precisions. ExoJAX returns float64 for float32 inputs when x64 is on: `voigt_exojax` casts back, so
   `voigt="harris"` keeps float32 under x64.
6. **Sums over many pieces.** Every term is positive, so the error of the sum is bounded by ~eps log2(N) of the sum
   (pairwise in XLA's reductions; the rows of `piece_chunk` pieces are summed last). Measured below.
7. **τ up to ~1e6 and exp(−τ).** float32 holds τ to 3.4e38 and exp(−τ) underflows to 0 above τ ≈ 104 (F < 1e-45), the
   same F as float64 to the precision of a spectrum. F = 1 − τ for τ < 6e-8: a float32 F of a very weak line is 1;
   compare such lines in τ (or cast τ to float64 before exp).
8. **Mixed inputs.** Host arrays (weights, offsets) arrive in float64 next to float32 fields: kernels take their working
   dtype from all their inputs (`summary_stats.stats_1d.p1d_kernel`). `voigt_convolution`, a standalone reference, works in float64 whenever x64 is on.

## float32 against float64

| quantity (TNG50-4, 4 skewers, 1 km/s pixels; the streaming route) | default mode | matched mode |
|---|---|---|
| Lyα: max \|ΔF\|, rms ΔF | 3.1e-6, 4.0e-7 | 1.7e-7, 1.9e-8 |
| Lyα: max \|Δτ/τ\| (τ > 1e-6 τ_max) | 2.8e-5 | 5.5e-7 |
| Si III: max \|ΔF\|, rms ΔF | 2.5e-7, 6.3e-9 | 2.0e-9, 3.7e-11 |
| Si III: max \|Δτ/τ\| (τ > 1e-6 τ_max) | 3.7e-5 | 1.3e-6 |

The same cells in both (the store is float32 in both, above): the residual is the τ sum's. In the default mode it is
the rounding of the float32 kernels (the offsets of the pixels from the line centres, the integrated profile); the
earlier default (midpoint sums of 0.25 km/s pieces) had 1.1e-5 / 1.7e-6 for Lyα. The matched mode forms its detunings
from float64 offsets on the host (`core.tau_engine.per_cell_line_tau`), hence its smaller residual in F. τ is returned
in the working dtype (float64 when the precision is float64).

## Tests

The suite runs in both precisions: `SKEWERJAX_PRECISION=float32 pytest` (the default) and `SKEWERJAX_PRECISION=float64
pytest`. A test that needs float64 (closed forms to 1e-10, bit-for-bit identities, the regression references) opts in
with `helpers.float64_only` and runs in float64 in both; `helpers.tol(f64, f32)` states a tolerance per precision.
