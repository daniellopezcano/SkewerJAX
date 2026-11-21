# The decomposition of the total overflux statistics

The two-point statistics of the total overflux δ_tot of L lines, in 3D on the lattice of skewers and in 1D, split
exactly into the contributions of pairs of subset fields, for any L: `skewerjax.summary_stats.decomposition`
(`decompose`, `decompose_fields`, `ensemble_decomposition`). The theory is in the review notes (`sec_flux.tex`,
Sec. Overflux; `sec_xi.tex`, Sec. Decomposition of the 2PCF; `sec_Pk.tex`); the equation labels below are theirs.

## 1. The fields

With δ_j = T_j / T̄_j − 1 the overflux of line j (`DeltaFieldBuilder`), A a subset of the lines, δ_A = ∏_{j∈A} δ_j and
𝒞 = T̄ / ∏_j T̄_j − 1 the correlator of the `mean` mode (one number for "global" or "box", a function of the pixel Λ for
"redshift", one row per skewer for "skewer"; Eq. overflux_denominator_expansion_general),

```
F_∅ = −𝒞 / (1 + 𝒞),      F_A = δ_A / (1 + 𝒞)  (|A| ≥ 1),      δ_tot = Σ_A F_A   exactly, at every pixel
```

(Eq. total_overflux_expansion_general; `overflux_check` reports the identity as `decomposition_error`). There are 2^L
fields: F_∅ (`ZERO_ORDER_KEY`, "𝒞") and one per non-empty subset.

## 2. Bilinearity

Every estimator of the package is a bilinear form of two fields: on the lattice, P_ab(k) = (dV/N) F_a(k) F_b(k)* and
ξ_ab(r) = (1/N) Σ_x a(x) b(x + r) (`fourier`), and every reduction of them is linear (the mean of P over the modes of a
bin, of ξ over the separations of a bin, the sum over k⊥ of P1D, the slice r⊥ = 0 of ξ1D). Hence, for every realisation,

```
S[δ_tot, δ_tot] = Σ_{A,B} S[F_A, F_B],     the order of (A, B) = |A| + |B| ∈ [0, 2L].
```

Exchanging A and B conjugates P and flips the lag (Eqs. Pk3D_cross_symmetry, xi3D_cross_symmetry), so the terms are
stored per unordered pair as the symmetric sums P_AB,sym = 2 Re P_AB and ξ_AB,sym(r) = ξ_AB(r) + ξ_AB(−r) (P_AA, ξ_AA for
A = B). The antisymmetric parts, Im P_AB (odd in k∥) and ξ_AB − ξ_BA (odd in Δ), carry the offsets of the lines: a
copy shifted by Δ₀ gives P_AB = e^{+ik∥Δ₀} P_AA. The binned 3D reductions fold the line of sight (|k∥|, |r∥|, μ ≥ 0),
where they cancel; they are returned along the line of sight (`P1D_imag`, `xi1D_anti`, signed lags).

## 3. The groupings

| key suffix | rows | content |
|---|---|---|
| `_order` | n = 0 … 2L | Σ of the pairs with \|A\| + \|B\| = n (the orders of the tables overflux_subset_pairings_L2, _L3) |
| `_pair` | the 2^(L−1)(2^L + 1) unordered pairs (`pairs`, `pair_orders`) | every subset pair, the tables xi1D_subset_pairings_L2/_L3_ensemble |
| `_o2` | `autos`, `crosses`, `CS2` | order 2: Ξ^(2) of Eq. Xi2_autos_crosses (Σ_j (j, j) and Σ_{i<j} (i, j)_sym) and its 𝒞 term Σ_{\|A\|=2} (∅, A)_sym |
| `_C` | `CC`, `CS1` … `CSL` | the 𝒞 terms: (∅, ∅) and Σ_{\|A\|=n} (∅, A)_sym, the −𝒞⟨S_n⟩ of Eq. xi1D_general_L_step1 |

`group="order"` (default) computes the orders, `_C` and `_o2` directly, without the pairs; `group="pair"` every pair;
`group="both"` the pairs and every grouping summed from them.

**The 𝒞 terms per realisation.** CS1 vanishes on average (⟨δ_j⟩ = 0); for one realisation it vanishes exactly only when
the means are those of the same skewers (then Σ_{x⊥} δ_j = 0 at every Λ), and otherwise not (`T_bar` of a larger
ensemble). When the means are those of the same skewers, Σ_{n≥2} CS_n = −2 CC as well. The notes collect them into
order 0: CC + Σ_n CS_n = −𝒞𝒞 / (1 + 𝒞)² on average (Eq. xi1D_general_L_decomposition). The orders here keep the
extremal pairs (∅, A) in order |A|; `notes_orders(record, R)` regroups them as the notes do (row 0 the collected 𝒞 terms,
row 1 zero, rows n ≥ 2 the pairs of non-empty subsets).

## 4. The reductions and the commuting diagram

The 3D reductions use the bins of `measure_lattice`, the `GridStatistics3D` of `lattice_grid`: every `GRID_KEYS`
(P isotropic, in the (k∥, k⊥) map, in μ wedges, the axis planes k⊥ = 0 and k∥ = 0; ξ isotropic, in the (r∥, r⊥) map,
in wedges), without the k = 0 mode, as `GridStatistics3D.measure`. The 1D reductions follow the commuting diagram of
the notes (Eq. Pk3D_diagram), term by term:

```
ξ_AB(r⊥, Δ)  ──Fourier──▶  P_AB(k⊥, k∥)
    │ r⊥ = 0                    │ L⊥⁻² Σ_{k⊥}        (Eq. Pk3D_P1D_from_P3D, its discrete form)
    ▼                           ▼
ξ1D_AB(Δ)    ──Fourier──▶  P1D_AB(k∥)               (Eq. Pk3D_P1D_from_xi1D)
```

P1D = L⊥⁻² Σ_{k⊥} P3D on the full k⊥ plane, which equals the mean over the skewers of the per-sightline cross spectra
(Parseval); ξ1D = irfft(P1D*)/dv, the r⊥ = 0 slice of ξ3D. The 1D reductions keep k = 0, as `stats_1d`; the 3D ones
drop it, so ξ1D = ξ3D(r⊥ = 0) + P(k = 0)/V, the product of the box means (`diagram_xi1D` reports the largest
difference of this identity over the terms). Units: 3D comoving (cMpc), 1D in velocity (k∥ in s/km, P1D in km/s, lags
in km/s), through u = dv / (L / n_los).

The dedicated 1D path, `decompose_1d` (one FFT per skewer, no lattice), stays the fast option; its orders and totals
equal those of the general decomposition to rounding (`tests/summary_stats/test_decomposition.py`).

**Conventions that differ from the notes** (the package's are kept): the notes' reductions in k⊥ and r⊥ integrate over
the angle (∫dφ, a factor 2π: ξ(r⊥ = 0, Δ) = 2π ξ(Δ) in Eq. xi3D_sample_zero_lag); here every bin is a mean over its modes
or separations, so the r⊥ = 0 column of the map is ξ1D itself. The notes' μ is signed in [−1, 1]; here μ = |μ|. With
a 𝒞(Λ), the factor 1/(1 + 𝒞)² of Eq. xi3D_decomposition_order2 is 1/[(1 + 𝒞(Λ))(1 + 𝒞(Λ + Δ))] inside the mean; the
fields F_A carry it pixel by pixel, so the decomposition stays exact.

## 5. Checks

`decompose` (from τ, with the total T / T̄ − 1 of `DeltaFieldBuilder`) or `decompose_fields(..., total=)` returns
`sum_rule`, per reduction: max |Σ of the terms − the total| / max(|total|, |terms|), the total by the existing estimators
(`measure_lattice` in 3D). The scale includes the terms because some totals cancel (P at k⊥ = 0 with the means of the same
skewers). It is ≲ 5e-7 in float32 (notebooks 10 and 11) and ≲ 1e-15 in float64 (toy lattices, L = 2 and 4).

**In the μ wedges** (`reductions=("P_mu", "xi_mu")`, the wedges of `GridStatistics3D`: `pk_mu_edges` for P, `xi_mu_edges`
for ξ, half-open, μ = 1 in the last), `wedge_sum_rule(record, "P_mu" | "xi_mu")` gives the same check wedge by wedge, each
relative to the largest |total| or |term| of its wedge (an `ensemble_decomposition`: the largest over the realisations).
A wedge is a mean over its modes, so the identity holds in every wedge and bin; the antisymmetric parts cancel there (the
wedges fold ±k∥). On the data of notebooks 10 and 11 (float32, F1): ≤ 3.6e-7 and ≤ 4.4e-7 per wedge of P (16 wedges),
≤ 3.4e-7 and ≤ 2.5e-7 per wedge of ξ (4 wedges). The figures: `plot_pk_mu_orders`, `plot_xi_mu_orders` (the orders, a
residual panel under each wedge) and `plot_pk_mu_terms`, `plot_xi_mu_terms` (each order term by term; `group="both"`).

## 6. Cost

2^L subset fields, one forward rfftn each; each term (a pair, or an order: its cross-powers summed first through
G_m = Σ_{|A|=m} F_A, so 2L + 1 terms, plus the L + 1 𝒞 terms and the split of order 2) is reduced as it is formed: the
binned P, one irfftn for the binned ξ (none if no ξ in 3D is asked, `reductions=`), and a k⊥ sum for P1D and ξ1D. No
2^L × 2^L array of 3D spectra is held. Memory: the 2^L transforms of the lattice plus a few grids
(`decomposition_memory(grid, L, group)`; `decompose_fields` raises `MemoryError` beyond the available memory).

## 7. Cost on the lattices of notebooks 10 and 11

Measured in the notebooks' kernels (float32, CPU, two lines, one realisation, `group="order"` unless said; the peak is the
largest increase of the process's resident memory during the call, sampled every 5 ms; 2026-11-20):

| lattice | `decompose`, first call (with compilation) | later calls | `group="both"` | peak memory | `decomposition_memory` | `decompose_1d` | `measure_lattice` (one field) |
|---|---|---|---|---|---|---|---|
| 10: 64 × 303 × 64 | 0.94 s | 0.49 s | 0.45 s | +73 MB (+206 MB with compilation) | 77 MB | 0.06 s | 0.04 s |
| 11: 64 × 128 × 64 | 0.65 s | 0.17 s | 0.17 s | +6 MB (+18 MB) | 33 MB | 0.02 s | 0.02 s |

Each term costs about one `measure_lattice` without its forward FFT (the binning dominates), so `group="order"` costs
about 3L + 4 of them and `group="pair"` 2^(L−1)(2^L + 1) (10 and 10 for two lines; 16 and 136 for four). The memory grows as
2^L times the lattice: by `decomposition_memory`, a 256² × 1024 lattice in float32 needs about 4.3 GB with two lines and
8.1 GB with four (`group="order"`).

The wedges alone (`reductions=("P_mu", "xi_mu")`, `group="both"`, one realisation, 2026-11-21): 10, 0.87 s (with
compilation) then 0.23 s, peak +169 MB (estimate 62 MB); 11, 0.68 s then 0.08 s, peak +23 MB (estimate 26 MB).
