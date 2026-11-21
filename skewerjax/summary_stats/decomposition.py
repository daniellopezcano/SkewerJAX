r"""The order-by-order decomposition of the two-point statistics of the total overflux, in 3D on the lattice of skewers
and in 1D, for any number of lines L (``decompose``, ``decompose_fields``, ``ensemble_decomposition``; ``docs/decomposition.md``).

**The fields** (review notes, Sec. Overflux, Eqs. delta_j, total_overflux_expansion_general). With delta_j the overflux of
line j, A a subset of the lines, delta_A = prod_{j in A} delta_j, and C = T_bar / prod_j T_bar_j - 1 the correlator of the
``mean`` mode (``overflux.DeltaFieldBuilder``: one number for "global" / "box", a function of the pixel Lambda for
"redshift", one row per skewer for "skewer"),

    F_{} = -C / (1 + C),    F_A = delta_A / (1 + C)  (|A| >= 1),    delta_tot = sum_A F_A  exactly, at every pixel

(the identity ``overflux_check`` reports as ``decomposition_error``; F_{} is ``ZERO_ORDER_KEY``, the notes' 𝒞 term).

**Bilinearity.** Every estimator here is a bilinear form S[a, b] of two fields: the lattice cross-power P_ab(k) =
(dV / N) F_a F_b^* and xi_ab(r) = (1/N) sum_x a(x) b(x + r) (``fourier``), and their reductions (bin means of P over modes,
of xi over separations, the k_perp sum of P1D, the r_perp = 0 slice of xi1D) are linear. So, for every realisation,

    S[delta_tot, delta_tot] = sum_{A, B} S[F_A, F_B]          (2^L x 2^L terms; the order of (A, B) is |A| + |B| in [0, 2L]).

Exchanging A and B conjugates P (P_BA = P_AB^*, notes Eq. Pk3D_cross_symmetry) and flips the lag (xi_BA(r) = xi_AB(-r),
Eq. xi3D_cross_symmetry), so the terms are stored as the symmetric pair sums over unordered pairs {A, B}:

    P_AB,sym = P_AB + P_BA = 2 Re P_AB,    xi_AB,sym(r) = xi_AB(r) + xi_BA(r) = xi_AB(r) + xi_AB(-r)   (A != B; P_AA, xi_AA).

The antisymmetric parts, Im P_AB (odd in k_par) and xi_AB - xi_BA (odd in Delta), carry the offsets of the lines (a
line shifted by Delta_0 gives P_AB = e^{+i k_par Delta_0} P_AA, Sec. Line-of-sight flip of the notes). Every binned 3D
reduction folds the line of sight (k_par = |k_los|, r_par = |r_los|, mu >= 0; ``GridStatistics3D``), where they cancel
(as in the notes' full-sphere monopole, Eq. Pk3D_mu_wedge_full); they are kept along the line of sight, in 1D
(``P1D_imag``, ``xi1D_anti``, signed lags).

**Groupings** (from the same terms; "pair" stores every pair, "order" only the sums below, without the pairs):

* by order n = |A| + |B|, n = 0 .. 2L (``*_order``): order 1 holds only (∅, j); order n >= 2 the "mid" pairs (A, B
  non-empty) and the "extremal" pairs (∅, A), |A| = n;
* by pair {A, B} (``*_pair``): the subset-pair tables of the notes (Tables overflux_subset_pairings_L2, _L3,
  xi1D_subset_pairings_L2_ensemble, _L3_ensemble);
* order 2 split as Xi^(2) of Eq. Xi2_autos_crosses plus its 𝒞 term (``*_o2``, rows ``O2_ROWS``): the autos
  sum_j (j, j), the crosses sum_{i<j} (i, j)_sym, and (∅, A)_sym with |A| = 2;
* the 𝒞 terms (``*_C``, rows ``C_ROWS``): CC = (∅, ∅) (order 0, +𝒞𝒞 / (1+𝒞)^2), and CS_n = sum_{|A|=n} (∅, A)_sym, n = 1 .. L (the
  notes' -𝒞 <S_n> of Eq. xi1D_general_L_step1). Per realisation CS_1 vanishes only on average, or exactly when the means
  are those of this realisation (then sum_{x_perp} delta_j = 0 at every Lambda); CC + sum_{n>=2} CS_n is the notes'
  collected order 0, -𝒞𝒞 / (1+𝒞)^2 on average (Eq. xi1D_general_L_decomposition: the notes move the extremal terms to
  order 0; here they stay in order |A|, and ``notes_orders`` regroups them).

**Reductions** (the same bins as ``measure_lattice``: the ``GridStatistics3D`` of ``lattice_grid``). 3D: every
``GRID_KEYS`` (P isotropic, the (k_par, k_perp) map, mu wedges, the axis planes k_perp = 0, k_par = 0; xi isotropic, the
(r_par, r_perp) map, wedges; "xi_ell" if the grid has multipoles), without the k = 0 mode, as ``GridStatistics3D.measure``.
1D, the commuting diagram of the notes (Eq. Pk3D_diagram): P1D = L_perp^-2 sum_{k_perp} P3D (Eq. Pk3D_P1D_from_P3D, its
discrete form) and xi1D = xi3D(r_perp = 0) (Eq. xi3D_sample_zero_lag) = irfft(P1D^*) / dv, with the k = 0 mode kept, as
``stats_1d`` (so xi1D = the r_perp = 0 slice of xi3D + P(k = 0) / V, the box mean the 3D statistics drop). Units:
3D comoving, as the grid ([cMpc^3], [1/cMpc], [cMpc]); 1D in velocity, k_par [s/km], P1D [km/s], lags [km/s]
(u = dv / (L / n_los)).

**Cost.** 2^L subset fields, one forward rfftn each; each term (a pair, or an order: the cross-powers of an order summed
first, through G_m = sum_{|A|=m} F_A) is reduced as it is formed: the binned P, one irfftn for the binned xi, and a k_perp
sum for P1D and xi1D. No 2^L x 2^L array of 3D spectra is held: the memory is the 2^L transforms plus a few grids
(``decomposition_memory``).

**Differences from the notes (the package's conventions are kept).** The notes' reductions integrate over the angle
(int d phi, Eqs. xi3D_sample_rpar_rperp, Pk3D_sample_kpar_kperp: a factor 2 pi, xi(r_perp = 0, Delta) = 2 pi xi(Delta)); here
every bin is a mean over its modes or separations, so the r_perp = 0 column of the map is xi1D itself. The notes' mu is
signed in [-1, 1]; here mu = |mu|. The 3D statistics drop k = 0 (the notes' integral constraint of a mean of the same
sample); the 1D ones keep it. With a 𝒞(Lambda) the factor 1 / (1 + 𝒞)^2 of Eq. xi3D_decomposition_order2 is, exactly,
1 / [(1 + 𝒞(Lambda)) (1 + 𝒞(Lambda + Delta))] inside the mean; the fields F_A carry it pixel by pixel.
"""

from __future__ import annotations

import logging
from functools import partial
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np

from skewerjax.summary_stats.binning import ensemble_mean
from skewerjax.summary_stats.fourier import lag_axis
from skewerjax.summary_stats.grid3d import GRID_KEYS, GridStatistics3D, _rfft_weights
from skewerjax.summary_stats.mesh import lattice_only
from skewerjax.summary_stats.orders import ZERO_ORDER_KEY, _C_rows
from skewerjax.summary_stats.overflux import DeltaFieldBuilder

logger = logging.getLogger(__name__)

GROUPS = ("order", "pair", "both")
REDUCTIONS_1D = ("P1D", "xi1D")
O2_ROWS = ("autos", "crosses", "CS2")  # order 2: sum_j (j, j); sum_{i<j} (i, j)_sym; sum_{|A|=2} (∅, A)_sym


def C_rows(n_lines: int) -> tuple:
    """The labels of the 𝒞 terms: "CC" = (∅, ∅), "CS<n>" = sum_{|A|=n} (∅, A)_sym, n = 1 .. L (module docstring)."""
    return ("CC",) + tuple(f"CS{n}" for n in range(1, int(n_lines) + 1))


def subset_terms(labels, key_joiner: str = ""):
    """The terms of the decomposition: the keys ([``ZERO_ORDER_KEY``] + every non-empty subset of ``labels``, single
    lines first, then subsets of 2, 3, ... lines, the order of ``DeltaFieldBuilder.subset_keys``), the members of each
    (tuples of line indices; () for ∅) and their orders |A|."""
    labels = list(labels)
    members = [()] + [m for r in range(1, len(labels) + 1) for m in combinations(range(len(labels)), r)]
    keys = [ZERO_ORDER_KEY] + [key_joiner.join(labels[i] for i in m) for m in members[1:]]
    return keys, members, [len(m) for m in members]


def decomposition_memory(grid: GridStatistics3D, n_lines: int, group: str = "order", dtype=jnp.float32) -> int:
    """An estimate of the peak memory [bytes] of ``decompose_fields`` on ``grid``: the 2^L transforms of the subset
    fields (+ the L + 1 sums G_m with ``group="order"``, + the total's), and about eight real grids of temporaries (a
    field, its transform, the term, its xi and the copies of the binning)."""
    real = np.dtype(jnp.zeros((), dtype).dtype).itemsize
    n_rfft = int(np.prod(grid._rfft_shape))
    n_c = 2**n_lines + 1 + (n_lines + 1 if group == "order" else 0)
    return int(n_c * 2 * real * n_rfft + 8 * real * grid.N)


def _available_memory() -> int | None:
    """MemAvailable of /proc/meminfo [bytes] (None where it cannot be read)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


@partial(jax.jit, static_argnames=("n_x", "n_z"))
def _lattice_fft(field, n_x: int, n_z: int):
    """rfftn of the fields of the skewers (N_sk, n_los), x-major (``lattice.on_lattice``), on the lattice (n_x, n_los,
    n_z): the layout of ``GridStatistics3D`` (rfft along the last, transverse, axis)."""
    return jnp.fft.rfftn(field.reshape(n_x, n_z, field.shape[-1]).swapaxes(-1, -2))


@jax.jit
def _kperp_sum(P, mirror):
    """The sum over the full k_perp plane of a spectrum P (n_x, n_los, K_z), in the rfft layout of the lattice (k_z >= 0),
    at k_par = 2 pi m / L, m = 0 .. n_los // 2: the stored modes (k_x, m, k_z), plus, for the k_z columns that stand for two
    modes (``mirror`` = 1), the modes (k_x, m, -k_z), P(k_x, m, -k_z) = conj P(-k_x, -m, k_z), summed over k_x.
    Returns (n_los // 2 + 1,)."""
    S = jnp.sum(P, axis=0)  # (n_los, K_z)
    n = S.shape[0]
    m = jnp.arange(n // 2 + 1)
    return jnp.sum(S[m], axis=-1) + jnp.conj(jnp.sum(S[(-m) % n] * mirror, axis=-1))


@lattice_only
def decompose_fields(
    delta_lines,
    C,
    grid: GridStatistics3D,
    dv: float,
    *,
    labels=None,
    key_joiner: str = "",
    total=None,
    group: str = "order",
    reductions=None,
    c_norm_min: float = 1e-12,
    check_memory: bool = True,
) -> dict:
    """The decomposition (module docstring) of the 3D and 1D two-point statistics of the total overflux of one
    realisation of a lattice of skewers.

    Parameters
    ----------
    delta_lines : (L, N_sk, n_los) the overflux delta_j of every line on the skewers, x-major (``lattice.on_lattice``).
    C : the correlator 𝒞 = T_bar / prod_j T_bar_j - 1 of the mean mode: a number, (n_los,) or (1, n_los) (shared by the
        skewers, e.g. ``DeltaFieldBuilder.C_per_pixel``), or (N_sk, n_los) (``mean="skewer"``); cast to the dtype of
        ``delta_lines`` (as ``OverfluxStatistics1D``).
    grid : the ``GridStatistics3D`` of the lattice (``lattice.lattice_grid``: line of sight along axis 1, shape (n_x,
        n_los, n_z), n_x n_z = N_sk): the 3D bins, those of ``measure_lattice``.
    dv : the pixel width [km/s] (the 1D reductions in velocity units).
    labels : one per line (default "1" .. "L"); key_joiner joins them into the subset keys.
    total : (N_sk, n_los), optional, the total overflux computed directly (T / T_bar - 1): its statistics (``*_total``;
        3D by ``GridStatistics3D.measure``, the existing estimator) and the sum rule (``sum_rule``).
    group : "order" (default): the orders, the 𝒞 terms, the split of order 2 (2L + 1 + L + 1 + 2 terms reduced);
        "pair": every unordered pair {A, B} (2^(L-1) (2^L + 1) terms); "both": the pairs and every grouping summed from them.
    reductions : the statistics to compute, any of ``GRID_KEYS`` (+ "xi_ell" with multipoles) and "P1D", "xi1D"
        (default: all). Without a key of xi in 3D no inverse 3D FFT is taken.
    c_norm_min : |1 + 𝒞| <= c_norm_min or a non-finite 𝒞 raises.
    check_memory : raise ``MemoryError`` if ``decomposition_memory`` exceeds the available memory.

    Returns
    -------
    dict of NumPy arrays (one record of ``ensemble_decomposition``):

    * axes and labels: ``k_par`` (n_los // 2 + 1,) [s/km], ``lags`` (n_los,) [km/s, centred], ``labels``, ``term_keys``
      (2^L,), ``term_orders``, ``orders`` (2L + 1,), ``C_rows`` (L + 1,), ``O2_rows`` (3,); with pairs ``pairs``
      (n_pairs, 2) str and ``pair_orders`` (n_pairs,); ``anti_pairs`` (n_anti, 2) (A != B: every pair with "pair" /
      "both", the single-line crosses (i, j), i < j, with "order");
    * for every reduction R (the shape of R: ``GRID_KEYS`` shapes of the grid, (n_los // 2 + 1,) for P1D, (n_los,) for
      xi1D): ``R_order`` (2L + 1, ...), ``R_C`` (L + 1, ...), ``R_o2`` (3, ...) with "order" / "both"; ``R_pair``
      (n_pairs, ...) with "pair" / "both"; ``R_total`` with ``total``;
    * the antisymmetric parts along the line of sight: ``P1D_imag`` (n_anti, n_los // 2 + 1) = Im P1D_AB [km/s],
      ``xi1D_anti`` (n_anti, n_los) = xi1D_AB - xi1D_BA (centred lags);
    * checks (floats): ``sum_rule`` {R: max |sum of the terms - R_total| / max(|R_total|, |terms|)} with ``total`` (the terms' scale
      where the total cancels, as P_kperp0 normalised by the mean of the same skewers);
      ``diagram_xi1D`` (with xi in 3D and xi1D): max over the terms of |xi1D - [xi3D(r_perp = 0) + P(k = 0) / V]| / max
      |xi1D|; ``memory_estimate`` [bytes].
    """
    if group not in GROUPS:
        raise ValueError(f"group must be one of {GROUPS}, got {group!r}.")
    d = jnp.asarray(delta_lines)
    if d.ndim != 3:
        raise ValueError(f"delta_lines must be (N_lines, N_sk, n_los), got shape {d.shape}.")
    n_lines, n_sk, n_los = d.shape
    if not isinstance(grid, GridStatistics3D) or grid.los_axis != 1:
        raise ValueError("grid must be the GridStatistics3D of the lattice (lattice_grid: the line of sight along axis 1).")
    n_x, n_g, n_z = grid.shape
    if n_g != n_los or n_x * n_z != n_sk:
        raise ValueError(f"the lattice {grid.shape} does not hold {n_sk} skewers of {n_los} pixels.")
    labels = [str(i + 1) for i in range(n_lines)] if labels is None else [str(x) for x in labels]
    if len(labels) != n_lines:
        raise ValueError(f"labels must have one entry per line ({n_lines}), got {len(labels)}.")
    keys_3d = GRID_KEYS + (("xi_ell",) if grid.multipoles else ())
    valid = keys_3d + REDUCTIONS_1D
    red = valid if reductions is None else tuple(reductions)
    bad = [r for r in red if r not in valid]
    if bad:
        raise ValueError(f"unknown reductions {bad}; valid: {valid}.")
    P3, X3 = [r for r in red if r.startswith("P_")], [r for r in red if r.startswith("xi_")]
    want_1d = [r for r in red if r in REDUCTIONS_1D]
    if not bool(jnp.all(jnp.isfinite(d))):
        raise ValueError("delta_lines has non-finite values: the lattice needs every skewer (drop or fix them first).")

    mem = decomposition_memory(grid, n_lines, group, d.dtype)
    avail = _available_memory() if check_memory else None
    if avail is not None and mem > avail:
        raise MemoryError(
            f"decompose_fields needs about {mem / 2**30:.1f} GiB ({2**n_lines} subset transforms of {grid._rfft_shape}), "
            f"{avail / 2**30:.1f} GiB available: use fewer lines, a smaller lattice, or check_memory=False."
        )

    Cr = _C_rows(C, (n_sk, n_los), d.dtype)
    if not bool(jnp.all(jnp.isfinite(Cr)) & jnp.all(jnp.abs(1.0 + Cr) > c_norm_min)):
        raise ValueError("The normalisation 1 + C is not finite or vanishes at some pixel.")
    w, c = 1.0 / (1.0 + Cr), Cr / (1.0 + Cr)  # as OverfluxStatistics1D: in the dtype of the fields

    keys, members, orders = subset_terms(labels, key_joiner)
    n_t = len(keys)
    fft = lambda f: _lattice_fft(f, n_x, n_z)
    F = [fft(jnp.broadcast_to(-c, (n_sk, n_los)))]  # F_{} = -C / (1 + C)
    for m in members[1:]:  # delta_A in the order of overflux_terms (left to right), then F_A = delta_A / (1 + C)
        term = d[m[0]]
        for j in m[1:]:
            term = term * d[j]
        F.append(fft(w * term))

    pref = grid.dV / grid.N
    A_perp = float(grid.L[0] * grid.L[2])
    u = float(dv) / float(grid.dx[1])  # [km/s per cMpc]: a pixel dv spans L / n_los
    mirror = jnp.asarray(_rfft_weights(n_z) - 1.0)
    shape = grid.shape
    diag = [0.0, 0.0]  # max |xi1D - slice|, max |xi1D|

    def p1d(P):  # (K,) [km/s], complex for a complex P
        return _kperp_sum(P, mirror) * (u / A_perp)

    def xi1d(P1):  # (n_los,) centred, from a (K,) P1D: irfft(P^*) / dv
        return jnp.fft.fftshift(jnp.fft.irfft(jnp.conj(P1), n=n_los) / dv)

    def reduce(p) -> dict:
        """Every requested reduction of a term given as its real, even per-mode spectrum p (rfft layout, k = 0 kept)."""
        out = grid._bin(p, None, keys=P3) if P3 else {}
        xi3 = None
        if X3:
            xi3 = jnp.fft.irfftn(p.at[0, 0, 0].set(0.0), s=shape) / grid.dV  # without k = 0, as GridStatistics3D.measure
            out.update(grid._bin(None, xi3, keys=X3))
        if want_1d:
            P1 = p1d(p).real
            x1 = xi1d(P1)
            if "P1D" in want_1d:
                out["P1D"] = np.asarray(P1)
            if "xi1D" in want_1d:
                out["xi1D"] = np.asarray(x1)
            if xi3 is not None and "xi1D" in want_1d:  # the commuting diagram: xi1D = xi3D(r_perp = 0) + P(k = 0) / V
                sl = jnp.fft.fftshift(xi3[0, :, 0] + p[0, 0, 0] / grid.V)
                diag[0] = max(diag[0], float(jnp.max(jnp.abs(x1 - sl))))
                diag[1] = max(diag[1], float(jnp.max(jnp.abs(x1))))
        return out

    def anti(P) -> dict:  # the antisymmetric parts along the line of sight of a complex P_AB
        P1 = p1d(P)
        return {"P1D_imag": np.asarray(P1.imag), "xi1D_anti": np.asarray(xi1d(P1) - xi1d(jnp.conj(P1)))}

    sym = lambda a, b: pref * (jnp.abs(a) ** 2 if a is b else 2.0 * (a * jnp.conj(b)).real)
    rows: dict = {}  # grouping -> list of reduction dicts

    def stack(name, reds):
        for r in red:
            rows.setdefault(f"{r}_{name}", np.stack([x[r] for x in reds]))

    singles = [i for i in range(n_t) if orders[i] == 1]
    out = dict(
        labels=np.array(labels, dtype=str),
        term_keys=np.array(keys, dtype=str),
        term_orders=np.array(orders),
        orders=np.arange(2 * n_lines + 1),
        C_rows=np.array(C_rows(n_lines), dtype=str),
        O2_rows=np.array(O2_ROWS, dtype=str),
        k_par=2 * np.pi * np.fft.rfftfreq(n_los, dv),
        lags=lag_axis(n_los, dv),
    )
    if group in ("pair", "both"):
        pairs = [(a, b) for a in range(n_t) for b in range(a, n_t)]
        reds = [reduce(sym(F[a], F[b])) for a, b in pairs]
        stack("pair", reds)
        anti_pairs = [(a, b) for a, b in pairs if a != b]
        po = np.array([orders[a] + orders[b] for a, b in pairs])
        out.update(pairs=np.array([(keys[a], keys[b]) for a, b in pairs], dtype=str), pair_orders=po)
        if group == "both":  # every grouping as sums of the pair rows (linear: the bins of a sum are the sums of the bins)
            for r in red:
                v = rows[f"{r}_pair"]
                rows[f"{r}_order"] = np.stack([v[po == n].sum(0) for n in range(2 * n_lines + 1)])
                cr = [[i for i, (a, b) in enumerate(pairs) if a == 0 and orders[b] == n] for n in range(n_lines + 1)]
                rows[f"{r}_C"] = np.stack([v[i].sum(0) for i in cr])
                au = [i for i, (a, b) in enumerate(pairs) if a == b and orders[a] == 1]
                cx = [i for i, (a, b) in enumerate(pairs) if a != b and orders[a] == 1 and orders[b] == 1]
                rows[f"{r}_o2"] = np.stack([v[au].sum(0), v[cx].sum(0), rows[f"{r}_C"][2] if n_lines >= 2 else 0 * v[0]])
    else:
        G = [sum(F[i] for i in range(n_t) if orders[i] == m) for m in range(n_lines + 1)]  # G_0 = F_{}
        by_order = []
        for n in range(2 * n_lines + 1):  # P_n = pref sum_{m + m' = n} G_m G_m'^*, real
            ms = [(m, n - m) for m in range(max(0, n - n_lines), n // 2 + 1)]
            by_order.append(reduce(sum(sym(G[a], G[b]) for a, b in ms)))
        stack("order", by_order)
        cc = [reduce(sym(G[0], G[0]))] + [reduce(sym(G[0], G[m])) for m in range(1, n_lines + 1)]
        stack("C", cc)
        autos = reduce(sum(sym(F[i], F[i]) for i in singles))
        crosses = [sym(F[i], F[j]) for i, j in combinations(singles, 2)]
        cross = reduce(sum(crosses)) if crosses else {r: 0 * v for r, v in autos.items()}
        stack("o2", [autos, cross, cc[2] if n_lines >= 2 else {r: 0 * v for r, v in autos.items()}])
        anti_pairs = list(combinations(singles, 2))
    out.update(rows)
    if anti_pairs:
        an = [anti(pref * F[a] * jnp.conj(F[b])) for a, b in anti_pairs]
        out["P1D_imag"] = np.stack([x["P1D_imag"] for x in an])
        out["xi1D_anti"] = np.stack([x["xi1D_anti"] for x in an])
    out["anti_pairs"] = np.array([(keys[a], keys[b]) for a, b in anti_pairs], dtype=str).reshape(-1, 2)

    if total is not None:
        t = jnp.asarray(total, d.dtype)
        if t.shape != (n_sk, n_los):
            raise ValueError(f"total must be (N_sk, n_los) = {(n_sk, n_los)}, got {t.shape}.")
        tot = grid.measure(t.reshape(n_x, n_z, n_los).swapaxes(-1, -2)) if (P3 or X3) else {}  # measure_lattice
        if want_1d:
            Ft = fft(t)
            P1 = p1d(pref * jnp.abs(Ft) ** 2).real
            tot.update(P1D=np.asarray(P1), xi1D=np.asarray(xi1d(P1)))
        top = "pair" if group == "pair" else "order"
        out["sum_rule"] = {}
        for r in red:
            out[f"{r}_total"] = np.asarray(tot[r])
            v, ref = out[f"{r}_{top}"], out[f"{r}_total"]
            scale = max(np.nanmax(np.abs(ref)), np.nanmax(np.abs(v)))  # a total that cancels (e.g. P_kperp0 at its own mean)
            out["sum_rule"][r] = float(np.nanmax(np.abs(v.sum(0) - ref)) / scale) if scale > 0 else 0.0
    if X3 and "xi1D" in want_1d:
        out["diagram_xi1D"] = diag[0] / diag[1] if diag[1] > 0 else 0.0
    out["memory_estimate"] = mem
    return out


@lattice_only
def decompose(
    tau,
    dv: float,
    mean_transmission=None,
    *,
    grid: GridStatistics3D,
    mean: str = "redshift",
    labels=None,
    group: str = "order",
    reductions=None,
    dtype=float,
    check_memory: bool = True,
) -> dict:
    """The decomposition of one realisation (``decompose_fields``) from the optical depths, with the overfluxes, 𝒞 and the
    total of ``DeltaFieldBuilder`` (the arguments of ``stats_1d.decompose_1d``, plus the lattice ``grid``).

    tau : (N_lines, N_sk, n_los), the skewers x-major on ``grid`` (``lattice_grid``); dv : the pixel width [km/s];
    mean_transmission : T_bar_j and T_bar of the ensemble, (N_lines + 1, n_los) or (N_lines + 1, N_sk, n_los) (None: the
    mean over these skewers, ``mean``); labels : one per line (default "1" .. "N_lines"); dtype : of the overflux fields
    (``DeltaFieldBuilder``; ``float``: float64 under x64). The total of the builder (T / T_bar
    - 1) gives the ``*_total`` and the sum rule. Returns the record of ``decompose_fields``."""
    Tb = None if mean_transmission is None else np.asarray(mean_transmission)
    if Tb is not None and Tb.ndim == 3 and (Tb.strides[1] == 0 or np.all(Tb == Tb[:, :1])):
        Tb = Tb[:, 0]  # shared by the skewers: one row per line (as decompose_1d)
    ov = DeltaFieldBuilder(tau, labels=None if labels is None else list(labels), dtype=dtype, mean=mean, mean_transmission=Tb)
    ov = ov.build()
    if ov.N_samples != np.shape(tau)[1]:
        raise ValueError("Some skewers were dropped (NaN in tau): the lattice needs every skewer.")
    delta = jnp.stack([ov.delta_terms[k] for k in ov.labels])
    return decompose_fields(
        delta,
        ov.C_per_pixel,
        grid,
        dv,
        labels=ov.labels,
        key_joiner=ov.key_joiner,
        total=ov.total,
        group=group,
        reductions=reductions,
        check_memory=check_memory,
    )


_AXES = (
    "labels",
    "term_keys",
    "term_orders",
    "orders",
    "C_rows",
    "O2_rows",
    "k_par",
    "lags",
    "pairs",
    "pair_orders",
    "anti_pairs",
)
_STAT_SUFFIXES = ("_order", "_pair", "_C", "_o2", "_total")


def ensemble_decomposition(records) -> dict:
    """Mean and error of the mean of ``decompose`` / ``decompose_fields`` records of independent realisations (as
    ``stats_1d.ensemble_1d``): the axes and labels of the first record and, for every statistic (``*_order``, ``*_pair``,
    ``*_C``, ``*_o2``, ``*_total``, ``P1D_imag``, ``xi1D_anti``), its mean, ``_err`` and ``_each``
    (``binning.ensemble_mean``); ``sum_rule`` and ``diagram_xi1D``: the largest over the realisations."""
    records = list(records)
    if not records:
        raise ValueError("ensemble_decomposition needs at least one record.")
    first = records[0]
    for r in records[1:]:
        if not (np.array_equal(r["term_keys"], first["term_keys"]) and r["k_par"].shape == first["k_par"].shape):
            raise ValueError("ensemble_decomposition: the records differ in their terms or their k grid.")
    stats = [k for k in first if k.endswith(_STAT_SUFFIXES) or k in ("P1D_imag", "xi1D_anti")]
    stats = [k for k in stats if k not in _AXES]
    out = {k: first[k] for k in _AXES if k in first}
    out.update(ensemble_mean(records, stats))
    if "sum_rule" in first:
        out["sum_rule"] = {k: max(r["sum_rule"][k] for r in records) for k in first["sum_rule"]}
    if "diagram_xi1D" in first:
        out["diagram_xi1D"] = max(r["diagram_xi1D"] for r in records)
    return out


WEDGE_REDUCTIONS = ("P_mu", "xi_mu")


def wedge_sum_rule(record, reduction: str = "P_mu", group: str | None = None) -> np.ndarray:
    """The sum rule of a decomposition, wedge by wedge: for every mu wedge of ``reduction`` ("P_mu" or "xi_mu", the
    wedges of ``GridStatistics3D``: ``pk_mu_edges``, ``xi_mu_edges``), max over its |k| or |r| bins of |sum of the terms
    - total| / max(|total|, |terms|) in that wedge (the scale of ``sum_rule``, per wedge; empty cells, NaN, skipped).

    record : a ``decompose`` / ``decompose_fields`` record with its total, or an ``ensemble_decomposition`` (then the
    largest over the realisations, from the ``_each`` arrays); group : "order" (default when the record has the orders)
    or "pair", the terms summed. Returns (n_mu,)."""
    if reduction not in WEDGE_REDUCTIONS:
        raise ValueError(f"reduction must be one of {WEDGE_REDUCTIONS}, got {reduction!r}.")
    group = group or ("order" if f"{reduction}_order" in record else "pair")
    if group not in ("order", "pair"):
        raise ValueError(f"group must be 'order' or 'pair', got {group!r}.")
    key, tot = f"{reduction}_{group}", f"{reduction}_total"
    if key not in record or tot not in record:
        raise ValueError(f"the record has no {key!r} or {tot!r} (decompose with a total and that grouping).")
    each = f"{key}_each" in record and f"{tot}_each" in record
    v = np.asarray(record[f"{key}_each"] if each else record[key][None], dtype=float)  # (n_real, n_terms, n_mu, n_bins)
    t = np.asarray(record[f"{tot}_each"] if each else record[tot][None], dtype=float)  # (n_real, n_mu, n_bins)
    diff = np.nanmax(np.abs(v.sum(1) - t), axis=-1)  # (n_real, n_mu)
    scale = np.maximum(np.nanmax(np.abs(t), axis=-1), np.nanmax(np.abs(v), axis=(1, -1)))
    return np.max(np.where(scale > 0, diff / np.where(scale > 0, scale, 1.0), 0.0), axis=0)


def notes_orders(record, reduction: str) -> np.ndarray:
    """The orders of a record grouped as the notes' Eq. xi1D_general_L_decomposition: row 0 the collected 𝒞 terms, CC +
    sum_n CS_n (-𝒞𝒞 / (1 + 𝒞)^2 on average), row 1 zero (order 1 is CS_1, folded into row 0), rows n >= 2 the "mid" pairs
    (A, B non-empty) of order n, order n - CS_n. ``reduction``: e.g. "P1D", "xi_map". Returns (2L + 1, ...)."""
    o, Cs = np.array(record[f"{reduction}_order"], dtype=float), np.asarray(record[f"{reduction}_C"])
    L = Cs.shape[0] - 1
    o[0] = Cs.sum(0)
    for n in range(1, L + 1):
        o[n] = o[n] - Cs[n]
    return o


__all__ = [
    "GROUPS",
    "O2_ROWS",
    "REDUCTIONS_1D",
    "C_rows",
    "subset_terms",
    "decomposition_memory",
    "decompose_fields",
    "decompose",
    "ensemble_decomposition",
    "notes_orders",
    "WEDGE_REDUCTIONS",
    "wedge_sum_rule",
]
