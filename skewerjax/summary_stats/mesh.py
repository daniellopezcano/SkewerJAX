"""Points on a periodic mesh: mass assignment, interlacing and the window correction, in D = 1, 2, 3 (JAX).

The Fourier field of a set of points with weights (e.g. the values of a field at the points, or unit weights for
discrete tracers) through a mesh of N_1 x ... x N_D nodes of spacing H = L / N per axis, after Sefusatti, Crocce,
Scoccimarro & Couchman 2016 (arXiv:1512.07295, "S16"; equation numbers counted from its LaTeX source):

* ``paint``: periodic deposition with the B-spline window of order p (S16 eqs. 15-18): every point goes to p nodes per
  axis, ``mas`` = "ngp" (p = 1), "cic" (2), "tsc" (3), "pcs" (4), weights w_j = prod_i W^(p)(x_i / H - j_i), summed over
  the points (one scatter-add of the p^D neighbours of every point); the weights of every point sum to one (mass
  conserved). ``shift`` (in units of H) displaces the points before the deposition (the interlaced mesh: 1/2).
* ``mas_window``: the window in Fourier space, W^(p)(k) = prod_i [sin(k_i H_i / 2) / (k_i H_i / 2)]^p (S16 eq. 27).
* ``fourier_mesh``: rfftn of the painted mesh; with ``interlace`` the average with the mesh of the points shifted by
  H/2 along every axis, brought back by the phase e^{+i sum_i k_i H_i / 2} (numpy's forward FFT is e^{-i k.x}: the
  shifted points sample the field at the nodes x_j - H/2), which cancels every alias image k - n k_s with odd
  n_1 + ... + n_D (S16 eqs. 33-36; k_s = 2 pi / H); with ``deconvolve`` divided by W(k) (S16 eq. 24). What remains is
  the alias sum of S16 eq. 37 (``aliased_power``).
* ``fourier_grid``: a field that already is the field on the mesh (grid-native): its rfftn, exact; no window and no
  interlacing apply (``mas`` = None in the statistics, ``GridStatistics3D.measure``).

Conventions: positions [cMpc] (any length unit, the same as L), wrapped periodically; the rfft layout (last axis
halved); k = 2 pi fftfreq(N, H) [1/cMpc]. Everything is jitted with ``n_mesh`` and ``mas`` static (no recompilation
for new positions, weights or L), and differentiable in the weights (and in the positions, where the kernel is
continuous: p >= 2). Memory: N_points p^D indices and weights (PCS in 3D: 64 per point).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

MAS_ORDERS = {"ngp": 1, "cic": 2, "tsc": 3, "pcs": 4}  # the B-spline order p of each mass-assignment scheme (S16 §2.3)
MAS_SCHEMES = tuple(MAS_ORDERS)


def mas_order(mas: str) -> int:
    """The order p of the mass-assignment scheme ``mas`` ("ngp", "cic", "tsc", "pcs"); raises on any other value."""
    if mas not in MAS_ORDERS:
        raise ValueError(f"mas must be one of {MAS_SCHEMES}, got {mas!r}.")
    return MAS_ORDERS[mas]


GRID_NATIVE_NOTE = (
    "Grid-native inputs (a field given on its own grid, or on the regular lattice of skewer samples) are the field on the "
    "mesh: their FFT is exact, with no window to divide and no alias images of an assignment to cancel, so ``mas`` must "
    "be None there (Sefusatti et al. 2016, §2.3: the window and its aliases arise only from assigning points to a mesh)."
)


def check_grid_native(mas, where: str) -> None:
    """Raise if a mass-assignment scheme is requested for a grid-native input (``mas`` must be None there)."""
    if mas is not None:
        raise ValueError(
            f"{where}: the input is already the field on its grid (grid-native): its FFT is exact and no mass "
            f"assignment applies, so mas must be None (got {mas!r}). Points take the point path "
            "(GridStatistics3D.measure_points, point_statistics)."
        )


def _geometry(n_mesh, L, D: int | None = None):
    """(n_mesh as a tuple of D ints, L as a float array (D,), H (D,))."""
    n = (int(n_mesh),) * (D or 1) if np.ndim(n_mesh) == 0 else tuple(int(m) for m in n_mesh)
    if D is not None and len(n) != D:
        raise ValueError(f"n_mesh {n_mesh} does not match the dimension D = {D} of the positions.")
    Ls = jnp.broadcast_to(jnp.asarray(L, dtype=float), (len(n),))
    return n, Ls, Ls / jnp.asarray(n, dtype=float)


def _axis_weights(u, p: int):
    """The first node (N,) and the weights (N, p) of the p nodes of every point along one axis, u the positions in
    units of H. Even p: the nodes floor(u) - p/2 + 1 .. floor(u) + p/2; odd p: centred on the nearest node."""
    if p % 2:
        i = jnp.floor(u + 0.5)
        s = u - i  # [-1/2, 1/2)
        if p == 1:
            w = jnp.ones_like(s)[:, None]
        else:  # TSC (S16 eq. 17)
            w = jnp.stack([0.5 * (0.5 - s) ** 2, 0.75 - s**2, 0.5 * (0.5 + s) ** 2], axis=-1)
        first = i - (p - 1) // 2
    else:
        i = jnp.floor(u)
        s = u - i  # [0, 1)
        if p == 2:  # CIC (S16 eq. 16)
            w = jnp.stack([1.0 - s, s], axis=-1)
        else:  # PCS (S16 eq. 18): distances 1 + s, s, 1 - s, 2 - s
            t = 1.0 - s
            w = jnp.stack(
                [t**3 / 6.0, (4.0 - 6.0 * s**2 + 3.0 * s**3) / 6.0, (4.0 - 6.0 * t**2 + 3.0 * t**3) / 6.0, s**3 / 6.0], axis=-1
            )
        first = i - (p // 2 - 1)
    return first.astype(jnp.int32), w


@partial(jax.jit, static_argnames=("n_mesh", "mas"))
def _paint(positions, weights, n_mesh, L, mas, shift):
    p = MAS_ORDERS[mas]
    D = len(n_mesh)
    H = L / jnp.asarray(n_mesh, dtype=L.dtype)
    u = positions / H + shift
    n_pts = positions.shape[0]
    flat, wt = None, None
    for a in range(D):
        first, w = _axis_weights(u[:, a], p)
        j = (first[:, None] + jnp.arange(p, dtype=jnp.int32)) % n_mesh[a]
        if flat is None:
            flat, wt = j, w
        else:
            flat = (flat[:, :, None] * n_mesh[a] + j[:, None, :]).reshape(n_pts, -1)
            wt = (wt[:, :, None] * w[:, None, :]).reshape(n_pts, -1)
    vals = weights[:, None] * wt
    mesh = jnp.zeros(int(np.prod(n_mesh)), dtype=vals.dtype).at[flat.ravel()].add(vals.ravel())
    return mesh.reshape(n_mesh)


def paint(positions, weights, n_mesh, L, mas: str = "pcs", shift: float = 0.0):
    """Periodic deposition of points with weights onto a mesh (module docstring; S16 eqs. 13-18).

    positions : (N, D) [cMpc] (D = 1, 2, 3; (N,) for D = 1), wrapped into the box; weights : (N,) (or a scalar);
    n_mesh : int or (D,) ints, the nodes per axis; L : the box side(s) [cMpc], scalar or (D,); mas : "ngp", "cic",
    "tsc", "pcs"; shift : a displacement of every point in units of the spacing H (1/2: the interlaced mesh).

    Returns the mesh (n_1, ..., n_D): the sum over the points of weight x W(x_j / H - node). Sum(mesh) = sum(weights)
    (to rounding). Node j sits at x = j H."""
    mas_order(mas)
    x = jnp.asarray(positions, dtype=float)
    x = x[:, None] if x.ndim == 1 else x
    if x.ndim != 2 or not 1 <= x.shape[1] <= 3:
        raise ValueError(f"positions must be (N, D) with D = 1, 2 or 3, got shape {tuple(np.shape(positions))}.")
    n, Ls, _ = _geometry(n_mesh, L, x.shape[1])
    w = jnp.broadcast_to(jnp.asarray(weights), (x.shape[0],))
    return _paint(x, w, n, Ls, mas, jnp.asarray(shift, dtype=x.dtype))


def mesh_k_axes(n_mesh, L, D: int | None = None) -> list:
    """The wavenumbers of the rfft layout of a mesh, one 1D array per axis [1/cMpc]: 2 pi fftfreq(n, H), the last axis
    2 pi rfftfreq(n, H)."""
    n, Ls, H = _geometry(n_mesh, L, D)
    axes = [2 * jnp.pi * jnp.fft.fftfreq(m, d=float(h)) for m, h in zip(n[:-1], np.asarray(H[:-1]))]
    axes.append(2 * jnp.pi * jnp.fft.rfftfreq(n[-1], d=float(H[-1])))
    return axes


def _outer(axes):
    """The per-axis arrays reshaped to broadcast as an ij grid."""
    D = len(axes)
    return [jnp.reshape(a, (1,) * i + (-1,) + (1,) * (D - i - 1)) for i, a in enumerate(axes)]


def mas_window(k, H, mas: str):
    """The Fourier window of the scheme ``mas``, W^(p)(k) = prod_i [sin(k_i H_i / 2) / (k_i H_i / 2)]^p (S16 eq. 27).

    k : (..., D) wave vectors [1/cMpc], or a list of D 1D arrays (the axes of a grid, e.g. ``mesh_k_axes``: the result
    broadcasts as an ij grid); H : the spacing(s) [cMpc], scalar or (D,)."""
    p = mas_order(mas)
    if isinstance(k, (list, tuple)):
        comps = _outer([jnp.asarray(a) for a in k])
        D = len(comps)
    else:
        kv = jnp.asarray(k)
        D = kv.shape[-1]
        comps = [kv[..., i] for i in range(D)]
    Hs = jnp.broadcast_to(jnp.asarray(H, dtype=float), (D,))
    out = 1.0
    for i, c in enumerate(comps):
        out = out * jnp.sinc(c * Hs[i] / (2 * jnp.pi)) ** p  # jnp.sinc(x) = sin(pi x) / (pi x)
    return out


def _interlace_phase(axes, H, sign: float = 1.0):
    """e^{sign i sum_i k_i H_i / 2} on the rfft grid of ``axes`` (sign +1: numpy's forward FFT, module docstring)."""
    comps = _outer(axes)
    arg = sum(c * (0.5 * H[i]) for i, c in enumerate(comps))
    return jnp.exp(sign * 1j * arg)


@partial(jax.jit, static_argnames=("n_mesh", "mas", "interlace", "deconvolve"))
def _fourier_mesh(positions, weights, n_mesh, L, mas, interlace, deconvolve, sign):
    D = len(n_mesh)
    H = L / jnp.asarray(n_mesh, dtype=L.dtype)
    axes = [2 * jnp.pi * jnp.fft.fftfreq(m) / H[i] for i, m in enumerate(n_mesh[:-1])]
    axes.append(2 * jnp.pi * jnp.fft.rfftfreq(n_mesh[-1]) / H[D - 1])
    F = jnp.fft.rfftn(_paint(positions, weights, n_mesh, L, mas, 0.0))
    if interlace:
        F2 = jnp.fft.rfftn(_paint(positions, weights, n_mesh, L, mas, 0.5))
        F = 0.5 * (F + _interlace_phase(axes, H, sign) * F2)
    if deconvolve:
        F = F / mas_window(axes, H, mas)
    return F


def fourier_mesh(positions, weights, n_mesh, L, *, mas: str = "pcs", interlace: bool = True, deconvolve: bool = True):
    """The Fourier field (rfft layout) of points with weights on a periodic mesh (module docstring; S16 §§2.3, 3.1).

    positions : (N, D) [cMpc]; weights : (N,); n_mesh : int or (D,); L : scalar or (D,) [cMpc]; mas : "ngp", "cic",
    "tsc", "pcs"; interlace : average with the mesh shifted by H/2 along every axis (the odd images cancel, S16 eq. 36);
    deconvolve : divide by ``mas_window`` (S16 eq. 24).

    Returns (F, meta): F = rfftn of the mesh of ``paint`` (the sum of the weights times the window: divide by the mean
    number of points per node for a field, e.g. the values of a field at the points), meta = dict(mas, interlace,
    deconvolve, n_mesh, L (D,), H (D,), k_nyq (D,) = pi / H)."""
    mas_order(mas)
    x = jnp.asarray(positions, dtype=float)
    x = x[:, None] if x.ndim == 1 else x
    if x.ndim != 2 or not 1 <= x.shape[1] <= 3:
        raise ValueError(f"positions must be (N, D) with D = 1, 2 or 3, got shape {tuple(np.shape(positions))}.")
    n, Ls, H = _geometry(n_mesh, L, x.shape[1])
    w = jnp.broadcast_to(jnp.asarray(weights), (x.shape[0],))
    F = _fourier_mesh(x, w, n, Ls, mas, bool(interlace), bool(deconvolve), 1.0)
    Hn = np.asarray(H)
    meta = dict(
        mas=mas, interlace=bool(interlace), deconvolve=bool(deconvolve), n_mesh=n, L=np.asarray(Ls), H=Hn, k_nyq=np.pi / Hn
    )
    return F, meta


def fourier_grid(field, L):
    """The Fourier field (rfft layout) of a field that already is the field on the mesh (grid-native): rfftn(field),
    exact, with no window and no interlacing (mas = None). Returns (F, meta) as ``fourier_mesh``."""
    f = jnp.asarray(field)
    n, Ls, H = _geometry(f.shape, L, f.ndim)
    Hn = np.asarray(H)
    meta = dict(mas=None, interlace=False, deconvolve=False, n_mesh=n, L=np.asarray(Ls), H=Hn, k_nyq=np.pi / Hn)
    return jnp.fft.rfftn(f), meta


def aliased_power(power, n_mesh, L, *, mas: str, interlace: bool, n_images: int = 3):
    """The expectation of the deconvolved spectrum of a field of spectrum ``power`` measured through ``fourier_mesh``
    (no shot noise), on the rfft modes of the mesh: the alias sum of S16 eqs. 29 and 37,

        P^G(k) = sum_n theta_n |W(k - n k_s) / W(k)|^2 P(|k - n k_s|),  theta_n = 1, or [1 + (-1)^{sum n}] / 2 interlaced,

    over the images |n_i| <= ``n_images`` per axis. ``power`` : callable P(|k|) [cMpc^D]; the k = 0 mode is set to 0."""
    p = mas_order(mas)
    n, Ls, H = _geometry(n_mesh, L)
    D = len(n)
    k = _outer(mesh_k_axes(n, Ls))
    ks = 2 * jnp.pi / H
    W0 = 1.0
    for i in range(D):
        W0 = W0 * jnp.sinc(k[i] * H[i] / (2 * jnp.pi)) ** p
    out = 0.0
    for img in np.ndindex(*(2 * n_images + 1,) * D):
        m = np.asarray(img) - n_images
        if interlace and int(m.sum()) % 2:
            continue
        q = [k[i] - m[i] * ks[i] for i in range(D)]
        Wq = 1.0
        for i in range(D):
            Wq = Wq * jnp.sinc(q[i] * H[i] / (2 * jnp.pi)) ** p
        qmod = jnp.sqrt(sum(c**2 for c in q))
        out = out + (Wq / W0) ** 2 * power(jnp.where(qmod > 0, qmod, 1.0)) * (qmod > 0)
    return jnp.asarray(out).at[(0,) * D].set(0.0)
