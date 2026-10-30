"""The order-by-order decomposition of the total overflux into its subset terms.

Decomposition (review notes, Eqs. total_overflux_expansion_general,
overflux_denominator_expansion_general, Table overflux_subset_pairings_L2, with the ensemble mean at
each pixel)
------------------------------------------------------------------------------
With ``summary_stats.overflux.DeltaFieldBuilder`` normalising every line and the total by its ensemble
mean transmission at each pixel v,

    delta_tot(v) = sum_A w(v) delta_A(v) - c(v),   w = 1/(1+C),  c = C/(1+C),

exactly, where A runs over all non-empty line subsets and C(v) = sum_{|A|>=2} <delta_A>_ens(v)
(= T_bar / prod_j T_bar_j - 1). The order-1 terms <delta_j>_ens vanish at every pixel because the
normalisation is ensemble-wide. The constant-across-samples -c(v) is the order-0 term (key
``ZERO_ORDER_KEY``). The term (A, B) has order |A| + |B|, and the sum of all weighted terms reproduces
the total P and xi exactly, at every k (including k = 0) and every lag.
"""

from __future__ import annotations

from functools import partial
from itertools import combinations
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

ZERO_ORDER_KEY = "𝒞"


def subset_orders(overflux, subset_keys: Sequence[str]) -> dict:
    """Order |A| (number of lines) of every subset key of a ``DeltaFieldBuilder``.

    Keys are regenerated from ``overflux.labels`` and ``overflux.key_joiner``; if those are
    missing, ``len(key)`` is used (single-character labels, as in the plotting tools).
    """
    labels = getattr(overflux, "labels", None)
    joiner = getattr(overflux, "key_joiner", "")
    if labels:
        orders = {joiner.join(m): r for r in range(1, len(labels) + 1) for m in combinations(labels, r)}
        if all(k in orders for k in subset_keys):
            return {k: orders[k] for k in subset_keys}
    return {k: len(k) for k in subset_keys}


def order_sum(pair_matrix, term_orders: Sequence[int]):
    """Sum a (n, n, ...) pair matrix into total orders |A|+|B| -> (2 max_order + 1, ...)."""
    o = np.asarray(term_orders)
    pair_ord = (o[:, None] + o[None, :]).ravel()
    n_ord = 2 * int(o.max()) + 1
    flat = pair_matrix.reshape((-1,) + pair_matrix.shape[2:])
    return jax.ops.segment_sum(flat, jnp.asarray(pair_ord), num_segments=n_ord)


@partial(jax.jit, static_argnames=("n_orders",))
def p3d_order_spectra(f_w, order_ids, pref, n_orders: int):
    """Weighted spectra summed per total order: P_n = pref sum_{m+m'=n} G_m G_m'^*,
    with G_m = sum_{|A|=m} F_w,A (G_0 = order-0 term). Returns (2 n_orders - 1, ...) complex."""
    G = jax.ops.segment_sum(f_w, order_ids, num_segments=n_orders)
    out = []
    for n_tot in range(2 * n_orders - 1):
        acc = jnp.zeros_like(G[0])
        for m in range(max(0, n_tot - n_orders + 1), min(n_tot, n_orders - 1) + 1):
            acc = acc + G[m] * jnp.conj(G[n_tot - m])
        out.append(pref * acc)
    return jnp.stack(out)
