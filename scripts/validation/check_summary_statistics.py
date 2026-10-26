#!/usr/bin/env python
"""Summary statistics: P1D / xi1D / P3D recovery of Gaussian random fields, the exact decomposition by order and line
pair, the shift signature of a second line, and the McDonald / Ma contamination fits (review notes, P1D, xi1D and
Sec. Overflux)."""
from _common import Checks, figure_backend

C = Checks("check_summary_statistics", __doc__)
from types import SimpleNamespace                                          # noqa: E402

import numpy as np                                                        # noqa: E402
import jax                                                                # noqa: E402
import jax.numpy as jnp                                                   # noqa: E402
from skewerjax.core import DeltaFieldBuilder                              # noqa: E402
from skewerjax.simulations import PowerLawGRF                             # noqa: E402
from skewerjax.core.stats import (OverfluxStatistics1D, OverfluxStatistics3D, fit_ma_model_profiled, fit_mcdonald_a,
                                  mcdonald_factor, xi_from_power)         # noqa: E402
from skewerjax.utils.plotting import hybrid_k_edges, pk_sigma             # noqa: E402

n_s_, Npix, L = (100 if C.quick else 500), 1000, 100.0
dv = L / Npix
shift = Npix // 10                                                        # second line at dv12 = L / 10
k_0, n_s = 2 * np.pi / 10.0, 0.5
A_s = 1.0 / float(PowerLawGRF(1.0, k_0, n_s, L=L, N=Npix).variance())     # sigma_delta = 1
grf = PowerLawGRF(A_s, k_0, n_s, L=L, N=Npix)
keys = jax.random.split(jax.random.PRNGKey(137), n_s_)
delta = jax.vmap(grf.sample)(keys)
single = lambda d: SimpleNamespace(subset_keys=["a"], delta_terms={"a": d}, C_per_sample=jnp.zeros((d.shape[0], 1)), total=d)
edges = hybrid_k_edges(2 * np.pi / L, grf.k_max[0])


def pulls(binned, expect):
    P_b, _, _, n_b = (np.asarray(x) for x in binned)
    s = pk_sigma(expect, n_b)
    ok = np.isfinite(P_b) & (n_b > 0) & np.isfinite(s)
    return (P_b[ok] - np.asarray(expect)[ok]) / s[ok]


# GRF level
st_g = OverfluxStatistics1D(single(delta), dv).compute_xi1d()
p = pulls(st_g.bin_p1d(edges), st_g.bin_p1d(edges, Pk=grf.power(st_g.k))[0])
C.check(f"GRF, {n_s_} skewers: binned P1D = input (chi2 / dof)", np.mean(p ** 2), 2.0, detail=f"max |pull| {np.max(np.abs(p)):.1f}")
xi_in = np.fft.fftshift(np.asarray(xi_from_power(grf.power(st_g.k), Npix, dv)))
xs = np.asarray(st_g.xi_total_per_sample)
xi_err = xs.std(0, ddof=1) / np.sqrt(xs.shape[0])
near = np.abs(np.asarray(st_g.lags)) <= 0.2 * L
C.check("GRF: xi1D = discrete inverse transform of the input (fraction of lags beyond 2 sigma, |dv| < L/5)",
        np.mean(np.abs(np.asarray(st_g.xi_total) - xi_in)[near] > 2 * xi_err[near]), 0.25)
# composite field: two lines, the second shifted by dv12
tau = jnp.stack([jnp.exp(delta), jnp.exp(jnp.roll(delta, shift, axis=-1))])
st = OverfluxStatistics1D(DeltaFieldBuilder(tau).build(), dv).compute_p1d().compute_xi1d()
sc = float(np.max(np.abs(np.asarray(st.Pk_total))))
C.check("decomposition, P1D: sum of the orders = total (max |diff| / max |P|)",
        np.max(np.abs(sum(np.asarray(v) for v in st.Pk_total_by_order.values()) - np.asarray(st.Pk_total))) / sc, 1e-10)
C.check("decomposition, xi1D: sum of the orders = total (max |diff| / max |xi|)",
        np.max(np.abs(sum(np.asarray(v) for v in st.xi_total_by_order.values()) - np.asarray(st.xi_total)))
        / float(np.max(np.abs(np.asarray(st.xi_total)))), 1e-10)
k = np.asarray(st.k)
P11, P12 = np.asarray(st.Pk_pair_complex[("1", "1")]), np.asarray(st.Pk_pair_complex[("1", "2")])
C.check("shift signature: P_12 = P_11 exp(+i k dv12) (max |diff| / max |P_11|)",
        np.max(np.abs(P12 - P11 * np.exp(1j * k * shift * dv))) / np.max(np.abs(P11)), 1e-10)
xi12 = np.asarray(st.xi_pair[("1", "2")])
C.check("shift signature: xi_12 peaks at +dv12 [km/s]", abs(np.argmax(xi12) * dv - shift * dv), dv)
# linear flux regime (sigma_delta = 0.01): P_tot -> tau0^2 (1 + a^2 + 2 a cos k dv12) P_in (McDonald form)
a_lin = 0.5
g_lin = PowerLawGRF(1e-4 * A_s, k_0, n_s, L=L, N=Npix)
d_lin = jax.vmap(g_lin.sample)(keys)
st_l = OverfluxStatistics1D(DeltaFieldBuilder(jnp.stack([jnp.exp(d_lin), a_lin * jnp.exp(jnp.roll(d_lin, shift, -1))])).build(), dv).compute_p1d()
k_l = np.asarray(st_l.k)
P_exp = mcdonald_factor(k_l, a_lin, delta_v=shift * dv) * np.asarray(g_lin.power(k_l))
e1 = hybrid_k_edges(2 * np.pi / L, grf.k_max[0], n_lin=Npix // 2, n_log=1)
p = pulls(st_l.bin_p1d(e1), st_l.bin_p1d(e1, Pk=P_exp)[0])
C.check("linear flux, two lines: P_tot = tau0^2 [1 + a^2 + 2a cos(k dv)] P_in (chi2 / dof)", np.mean(p ** 2), 2.0)
fit = fit_mcdonald_a(k_l[1:], np.asarray(st_l.Pk_total)[1:], np.asarray(st_l.Pk_pair[("1", "1")])[1:], delta_v=shift * dv,
                     k_range=(0.0, 1.0))
C.check("McDonald fit recovers the input a (relative error)", abs(fit["a"] / a_lin - 1), 0.02)
fit_ma = fit_ma_model_profiled(k_l[1:], np.asarray(st_l.Pk_total)[1:], np.asarray(st_l.Pk_pair[("1", "1")])[1:], delta_v=shift * dv,
                               k_range=(0.0, 1.0), run_global=False)
C.check("Ma et al. chi2 <= McDonald chi2 (nested model; ratio - 1)", fit_ma["chi2"] / fit["chi2"] - 1, 1e-9)
# 3D GRF: P3D(|k|) recovery on an isotropic grid
n3 = 16 if C.quick else 32
g3 = PowerLawGRF(1.0, 1.0, -1.0, L=(20.0,) * 3, N=(n3,) * 3)
d3 = g3.sample(jax.random.PRNGKey(3))
st3 = OverfluxStatistics3D(single(d3.reshape(n3 * n3, n3)), 20.0 / n3, 20.0 / n3).compute_p3d()
b3 = st3.bin_p3d_kmod(n_bins=10)
p = pulls(b3, st3.bin_p3d_kmod(k_edges=b3[2], Pk3d_input=g3.power(jnp.sqrt(st3.k_perp_2d[:, :, None] ** 2 + st3.k_par[None, None, :] ** 2)))[0])
C.check(f"GRF 3D ({n3}^3): binned P3D(|k|) = input (chi2 / dof)", np.mean(p ** 2), 2.0)

if C.outdir is not None:
    figure_backend()
    from skewerjax.utils.plotting import plot_orders_superimposed, plot_pk_recovery
    fig, _ = plot_pk_recovery(st_g.bin_p1d(edges), st_g.bin_p1d(edges, Pk=grf.power(st_g.k))[0], power=grf.power,
                              label=f"GRF, {n_s_} skewers", k_label="$k$", P_label="$P(k)$")
    C.savefig(fig, "stats_p1d_recovery")
    fig, _ = plot_orders_superimposed(st, "P")
    C.savefig(fig, "stats_p1d_orders")
C.done()
