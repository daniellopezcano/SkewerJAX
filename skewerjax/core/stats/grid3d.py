"""Two-point statistics of a periodic 3D field on its own grid, and their exact expectations (notebook 08, Part A).

One realisation delta on an (N_x, N_y, N_z) grid of a box of sides L, the line of sight along ``los_axis``:

    P_hat(k)  = |delta_k|^2 / V = (dV / N) |FFT(delta)|^2            (``simulations.PowerLawGRF``'s normalisation)
    xi_hat(r) = (1/N) sum_x delta(x) delta(x + r) = irfftn(P_hat) / dV  (Wiener-Khinchin; the exact discrete pair)

(``two_point``'s conventions with d = 3; r runs over the minimum-image separations of the grid). Both are averaged in
bins: P over the modes of a bin (each rfft mode counted with its half-space multiplicity), xi over the separation
vectors of a bin, in three ways each:

* isotropic: |k| or |r|;
* the plane (k_par, k_perp) or (r_par, r_perp), with k_par = |k_los| and k_perp = |k| across the line of sight;
* wedges of mu (|k| or |r| in each interval of mu = k_par / |k| or r_par / r; ``MU_WEDGES``);

and, optionally (``multipoles=True``), the Legendre multipoles xi_0, xi_2, xi_4 of xi in the |r| bins of the wedges: in
each bin, the least-squares fit xi(r) = sum_l xi_l L_l(mu) over its separation vectors, so the monopole xi_0 does not
depend on how the separations of a bin are spread in mu (on a coarse lattice, e.g. of skewers, the plain |r| average of
an anisotropic xi does).

**Expectations.** For a Gaussian field with spectrum P(k) on the grid's modes, <P_hat(k)> = P(k) at every mode and
<xi_hat(r)> = xi_in(r) = (1/V) sum_k P(k) e^{i k.r} = irfftn(P on the modes) / dV at every separation (no k = 0 mode).
``GridStatistics3D.theory`` bins these exactly like the measurement, so the binned expectation is exact, free of the
variation of P or xi inside a bin (the BAO wiggles included); ``GridStatistics3D.xi_iso_covariance`` gives the exact
Gaussian covariance of the isotropic xi bins of one realisation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from skewerjax.core.stats.two_point import bin_index, binned_mean

MU_WEDGES = (0.0, 0.5, 0.8, 0.95, 1.0)  # mu = r_par / r: the wedges of DESI DR2 Lya BAO (arXiv:2503.14739, Fig. 3)
GRID_KEYS = ("P_iso", "P_map", "P_mu", "xi_iso", "xi_map", "xi_mu")


def hybrid_k_edges(k_fund, k_max, n_lin: int = 12, n_log: int = 18):
    """k bin edges: one mode per bin up to (n_lin + 1/2) k_fund (edges at (j + 1/2) k_fund), then n_log log bins."""
    lin = (np.arange(n_lin + 1) + 0.5) * k_fund
    lin = lin[lin < k_max]
    return np.concatenate([lin[:-1], np.geomspace(lin[-1], k_max, n_log + 1)])


def occupied_edges(values, edges):
    """The bin edges with every empty bin merged into the next one, so that each bin holds at least one of ``values``
    (a log bin narrower than the spacing of the discrete modes along an axis would be empty: a blank row or column)."""
    v = np.unique(np.asarray(values, dtype=float))
    out = [edges[0]]
    for hi in edges[1:]:
        if np.any((v >= out[-1]) & (v < hi)):
            out.append(hi)
    out[-1] = edges[-1]  # trailing empty bins join the last occupied one
    return np.asarray(out)


def _rfft_weights(n_last: int) -> np.ndarray:
    """Full-grid modes represented by each column of the last rfft axis: 2, except k = 0 and Nyquist (1)."""
    w = np.full(n_last // 2 + 1, 2.0)
    w[0] = 1.0
    if n_last % 2 == 0:
        w[-1] = 1.0
    return w


def _flat2(i, j, n_j):
    """Flat index of two bin indices (-1 if either is outside)."""
    return np.where((i >= 0) & (j >= 0), i * n_j + j, -1)


class GridStatistics3D:
    """P and xi of periodic 3D fields on one grid, binned three ways (module docstring); bins built once, reused.

    Parameters
    ----------
    shape : (N_x, N_y, N_z); L : box side [cMpc] (scalar or (3,)); los_axis : the axis of the line of sight.
    k_edges : |k| bins [1/cMpc] (default: one mode per bin for the first 12 fundamental modes, then 24 log bins to the
        Nyquist frequency of the coarsest axis); kpar_edges, kperp_edges : the (k_par, k_perp) plane (default: 6 linear
        and 20 log bins, empty rows or columns merged, ``occupied_edges``); k_mu_edges : |k| bins of the wedges (default:
        16 log bins).
    r_edges : |r| bins [cMpc] (default: width 3 cells up to L / 2); rpar_edges, rperp_edges : the (r_par, r_perp)
        plane (default: the same edges); r_mu_edges : |r| bins of the wedges (default: ``r_edges``, r = 0 excluded).
    mu_edges : the wedges (default ``MU_WEDGES``; the last edge inclusive).
    multipoles : also measure the multipoles l = 0, 2, 4 of xi (key "xi_ell", (3, n_rmu)), NaN in bins whose separations
        do not span enough values of mu to separate them (relative condition number of the fit > 1e8).

    Attributes after construction: the bin centres (mode- or separation-weighted means) ``k_iso`` (n_k,), ``kpar``,
    ``kperp`` (the centres of the map's rows and columns), ``k_mu`` (n_mu, n_kmu); ``r_iso``, ``rpar``, ``rperp``,
    ``r_mu``; the counts ``n_modes`` / ``n_lags`` per binning (dicts keyed like ``GRID_KEYS``, the half-space weighted
    number of modes for P, the number of separation vectors for xi).
    """

    def __init__(
        self,
        shape,
        L,
        *,
        los_axis: int = 1,
        k_edges=None,
        kpar_edges=None,
        kperp_edges=None,
        k_mu_edges=None,
        r_edges=None,
        rpar_edges=None,
        rperp_edges=None,
        r_mu_edges=None,
        mu_edges=MU_WEDGES,
        multipoles: bool = False,
    ):
        self.shape = tuple(int(n) for n in shape)
        if len(self.shape) != 3:
            raise ValueError(f"GridStatistics3D needs a 3D grid, got shape {shape}.")
        self.L = np.broadcast_to(np.asarray(L, dtype=float), (3,)).copy()
        self.los_axis = int(los_axis) % 3
        self.dx = self.L / np.asarray(self.shape)
        self.N = int(np.prod(self.shape))
        self.dV = float(np.prod(self.dx))
        self.V = float(np.prod(self.L))
        k_f, k_nyq = 2 * np.pi / self.L.max(), float(np.min(np.pi / self.dx))
        r_max = 0.5 * float(self.L.min())
        self.mu_edges = np.asarray(mu_edges, dtype=float)
        n_mu = self.mu_edges.size - 1

        # ---- modes (rfft layout, last axis halved): |k|, k_par, k_perp, mu, half-space weights
        axes = [2 * np.pi * np.fft.fftfreq(n, d) for n, d in zip(self.shape[:-1], self.dx[:-1])]
        axes.append(2 * np.pi * np.fft.rfftfreq(self.shape[-1], self.dx[-1]))
        kv = np.meshgrid(*axes, indexing="ij", sparse=True)
        kmod = np.sqrt(sum(k**2 for k in kv)).ravel()
        kpar = np.broadcast_to(np.abs(kv[self.los_axis]), tuple(len(a) for a in axes)).ravel()
        kperp = np.sqrt(np.maximum(kmod**2 - kpar**2, 0.0))
        live = kmod > 0
        mu_k = np.where(live, kpar / np.where(live, kmod, 1.0), 0.0)
        w = np.broadcast_to(_rfft_weights(self.shape[-1]), tuple(len(a) for a in axes)).ravel()
        self._w_modes = jnp.asarray(w)

        self.k_edges = hybrid_k_edges(k_f, k_nyq, 12, 24) if k_edges is None else np.asarray(k_edges, float)
        e2 = hybrid_k_edges(k_f, k_nyq, 6, 20)
        self.kpar_edges = occupied_edges(kpar[live], e2) if kpar_edges is None else np.asarray(kpar_edges, float)
        self.kperp_edges = occupied_edges(kperp[live], e2) if kperp_edges is None else np.asarray(kperp_edges, float)
        self.k_mu_edges = np.geomspace(k_f, k_nyq, 17) if k_mu_edges is None else np.asarray(k_mu_edges, float)
        nkm = self.k_mu_edges.size - 1
        npar, nperp = self.kpar_edges.size - 1, self.kperp_edges.size - 1
        idx_k = {
            "iso": np.where(live, bin_index(kmod, self.k_edges), -1),
            "map": np.where(live, _flat2(bin_index(kpar, self.kpar_edges), bin_index(kperp, self.kperp_edges), nperp), -1),
            "mu": np.where(live, _flat2(bin_index(mu_k, self.mu_edges), bin_index(kmod, self.k_mu_edges), nkm), -1),
        }
        self._nb = {"iso": self.k_edges.size - 1, "map": npar * nperp, "mu": n_mu * nkm}
        self._shape_k = {"iso": (self._nb["iso"],), "map": (npar, nperp), "mu": (n_mu, nkm)}
        self._idx_k = {b: jnp.asarray(i.astype(np.int32)) for b, i in idx_k.items()}

        # ---- separations (the full lag grid, minimum image): |r|, r_par, r_perp, mu
        off = [d * np.where(np.arange(n) <= n // 2, np.arange(n), np.arange(n) - n) for n, d in zip(self.shape, self.dx)]
        rv = np.meshgrid(*off, indexing="ij", sparse=True)
        rmod = np.sqrt(sum(r**2 for r in rv)).ravel()
        rpar = np.broadcast_to(np.abs(rv[self.los_axis]), self.shape).ravel()
        rperp = np.sqrt(np.maximum(rmod**2 - rpar**2, 0.0))
        mu_r = np.where(rmod > 0, rpar / np.where(rmod > 0, rmod, 1.0), 0.0)
        dr = 3 * float(self.dx.max())
        self.r_edges = np.arange(0.0, r_max + 1e-9 * r_max, dr) if r_edges is None else np.asarray(r_edges, float)
        self.rpar_edges = self.r_edges if rpar_edges is None else np.asarray(rpar_edges, float)
        self.rperp_edges = self.r_edges if rperp_edges is None else np.asarray(rperp_edges, float)
        self.r_mu_edges = self.r_edges if r_mu_edges is None else np.asarray(r_mu_edges, float)
        nrm = self.r_mu_edges.size - 1
        nrpar, nrperp = self.rpar_edges.size - 1, self.rperp_edges.size - 1
        idx_r = {
            "iso": bin_index(rmod, self.r_edges),
            "map": _flat2(bin_index(rpar, self.rpar_edges), bin_index(rperp, self.rperp_edges), nrperp),
            "mu": np.where(rmod > 0, _flat2(bin_index(mu_r, self.mu_edges), bin_index(rmod, self.r_mu_edges), nrm), -1),
        }
        self._nb.update({"r_iso": self.r_edges.size - 1, "r_map": nrpar * nrperp, "r_mu": n_mu * nrm})
        self._shape_r = {"iso": (self.r_edges.size - 1,), "map": (nrpar, nrperp), "mu": (n_mu, nrm)}
        self._idx_r = {b: jnp.asarray(i.astype(np.int32)) for b, i in idx_r.items()}
        self._ones = jnp.ones(rmod.size)
        self.multipoles = bool(multipoles)
        if self.multipoles:  # Legendre L_0, L_2, L_4 of every separation, and the normal matrix of every |r| bin
            leg = np.stack([np.ones_like(mu_r), 0.5 * (3 * mu_r**2 - 1), (35 * mu_r**4 - 30 * mu_r**2 + 3) / 8])
            self._idx_ell = jnp.asarray(np.where(rmod > 0, bin_index(rmod, self.r_mu_edges), -1).astype(np.int32))
            self._leg = jnp.asarray(leg)
            M = np.stack(
                [
                    np.asarray(binned_mean(jnp.asarray(leg[i] * leg[j]), self._idx_ell, self._ones, nrm)[0])
                    for i in range(3)
                    for j in range(3)
                ]
            )
            M = np.moveaxis(M.reshape(3, 3, nrm), -1, 0)
            ok = np.all(np.isfinite(M), axis=(1, 2))
            ok[ok] &= np.linalg.cond(M[ok]) < 1e8
            self._ell_inv = np.where(ok[:, None, None], np.linalg.inv(np.where(ok[:, None, None], M, np.eye(3))), np.nan)

        # ---- bin centres and counts
        def centres(x, idx, wt, shp):
            m, W = binned_mean(jnp.asarray(x), idx, wt, int(np.prod(shp)))
            return np.asarray(m).reshape(shp), np.asarray(W).reshape(shp)

        self.k_iso, n_iso = centres(kmod, self._idx_k["iso"], self._w_modes, self._shape_k["iso"])
        kpar_c, n_map = centres(kpar, self._idx_k["map"], self._w_modes, self._shape_k["map"])
        kperp_c, _ = centres(kperp, self._idx_k["map"], self._w_modes, self._shape_k["map"])
        self.kpar = np.sqrt(self.kpar_edges[1:] * self.kpar_edges[:-1])  # the geometric centres of the map's axes
        self.kperp = np.sqrt(self.kperp_edges[1:] * self.kperp_edges[:-1])
        self.kpar_map, self.kperp_map = kpar_c, kperp_c  # the mode-weighted centre of every cell of the map
        self.k_mu, n_kmu = centres(kmod, self._idx_k["mu"], self._w_modes, self._shape_k["mu"])
        self.r_iso, c_iso = centres(rmod, self._idx_r["iso"], self._ones, self._shape_r["iso"])
        self.rpar = 0.5 * (self.rpar_edges[1:] + self.rpar_edges[:-1])
        self.rperp = 0.5 * (self.rperp_edges[1:] + self.rperp_edges[:-1])
        _, c_map = centres(rmod, self._idx_r["map"], self._ones, self._shape_r["map"])
        self.r_mu, c_mu = centres(rmod, self._idx_r["mu"], self._ones, self._shape_r["mu"])
        self.n_modes = {"P_iso": n_iso, "P_map": n_map, "P_mu": n_kmu}
        self.n_lags = {"xi_iso": c_iso, "xi_map": c_map, "xi_mu": c_mu}

    # ------------------------------------------------------------------ binning
    def _bin(self, P_modes, xi_lags) -> dict:
        P_modes, xi_lags = jnp.ravel(P_modes), jnp.ravel(xi_lags)
        out = {}
        for b in ("iso", "map", "mu"):
            P_b, _ = binned_mean(P_modes, self._idx_k[b], self._w_modes, int(np.prod(self._shape_k[b])))
            xi_b, _ = binned_mean(xi_lags, self._idx_r[b], self._ones, int(np.prod(self._shape_r[b])))
            out["P_" + b] = np.asarray(P_b).reshape(self._shape_k[b])
            out["xi_" + b] = np.asarray(xi_b).reshape(self._shape_r[b])
        if self.multipoles:
            n = self._ell_inv.shape[0]
            v = np.stack([np.asarray(binned_mean(L * xi_lags, self._idx_ell, self._ones, n)[0]) for L in self._leg])
            out["xi_ell"] = np.einsum("bij,jb->ib", self._ell_inv, np.nan_to_num(v))
        return out

    def measure(self, delta) -> dict:
        """The binned P and xi of one realisation ``delta`` (``shape``): dict keyed by ``GRID_KEYS`` (P [cMpc^3], xi
        dimensionless; NaN in empty bins). One rfftn and one irfftn."""
        d = jnp.asarray(delta)
        if d.shape != self.shape:
            raise ValueError(f"delta must have the grid shape {self.shape}, got {d.shape}.")
        f2 = jnp.abs(jnp.fft.rfftn(d)) ** 2
        P = (self.dV / self.N) * f2
        xi = jnp.fft.irfftn(f2, s=self.shape) / self.N
        return self._bin(P, xi)

    def theory(self, power_modes) -> dict:
        """The exact expectations of ``measure`` for a Gaussian field with spectrum ``power_modes`` on the rfft modes of
        the grid (e.g. ``PowerLawGRF.power_grid()``): the spectrum binned like the measurement, and xi_in =
        irfftn(P) / dV, the correlation of the discrete periodic field, binned alike (module docstring)."""
        P = jnp.asarray(power_modes)
        if P.shape != self._rfft_shape:
            raise ValueError(f"power_modes must have the rfft shape {self._rfft_shape}, got {P.shape}.")
        return self._bin(P, jnp.fft.irfftn(P, s=self.shape) / self.dV)

    def xi_iso_covariance(self, power_modes) -> np.ndarray:
        """The exact covariance (n_r, n_r) of the isotropic xi bins of one realisation of a Gaussian field with spectrum
        ``power_modes`` (rfft layout, as ``theory``); the error of the mean of n independent realisations has C / n.

        xi_hat(r) = (1/V) sum_k P_hat(k) cos(k.r) over the full grid, and the periodogram of a Gaussian field has
        Cov[P_hat(k), P_hat(k')] = P(k)^2 (delta_{k,k'} + delta_{k,-k'}), so

            Cov[xi_hat(r), xi_hat(r')] = [S(r - r') + S(r + r')] / V^2,    S(u) = sum_k P(k)^2 e^{i k.u},

        and, averaged over the separations of bins a and b (each symmetric under r -> -r),
        C_ab = 2 / (V^2 n_a n_b) sum_{r in a} (S * 1_b)(r), with 1_b the indicator of bin b on the lag grid: one
        convolution (an rfftn and an irfftn) per bin. NaN for empty bins."""
        P2 = jnp.asarray(power_modes) ** 2
        if P2.shape != self._rfft_shape:
            raise ValueError(f"power_modes must have the rfft shape {self._rfft_shape}, got {P2.shape}.")
        idx = np.asarray(self._idx_r["iso"])
        nb = self._shape_r["iso"][0]
        cnt = np.asarray(self.n_lags["xi_iso"], dtype=float)
        ok = idx >= 0
        conv = jax.jit(lambda ind: jnp.fft.irfftn(P2 * jnp.fft.rfftn(ind), s=self.shape).ravel() * self.N)
        C = np.empty((nb, nb))
        for b in range(nb):
            s = np.asarray(conv(jnp.asarray((idx == b).reshape(self.shape), dtype=P2.dtype)))
            C[:, b] = np.bincount(idx[ok], weights=s[ok], minlength=nb)
        n = np.where(cnt > 0, cnt, np.nan)
        C = 2.0 * C / (self.V**2 * np.outer(n, n))
        return 0.5 * (C + C.T)

    @property
    def _rfft_shape(self) -> tuple:
        return self.shape[:-1] + (self.shape[-1] // 2 + 1,)

    def axes(self) -> dict:
        """The bin edges and centres (NumPy), e.g. to store them with the measurements."""
        names = ("k_edges", "kpar_edges", "kperp_edges", "k_mu_edges", "r_edges", "rpar_edges", "rperp_edges")
        out = {n: getattr(self, n) for n in names + ("r_mu_edges", "mu_edges")}
        for n in ("k_iso", "kpar", "kperp", "kpar_map", "kperp_map", "k_mu", "r_iso", "rpar", "rperp", "r_mu"):
            out[n] = getattr(self, n)
        out.update({"n_modes_" + k: v for k, v in self.n_modes.items()})
        out.update({"n_lags_" + k: v for k, v in self.n_lags.items()})
        return out
