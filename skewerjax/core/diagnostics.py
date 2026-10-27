"""Debugging tools: what to call when a result looks wrong.

* ``check_skewers(cells, skewer_bundle, line_bundle)``: independent checks of the route cells -> pieces -> tau for one
  set of skewers, returned as a ``SkewerCheck`` (``print(report)`` gives a summary, ``report.ok`` the verdict):

  1. **coverage**: the pieces tile every path, sum_p l_p = path length (exact up to round-off for the Voronoi
     depositions; skipped for the top-hat spheres, which overlap);
  2. **column**: sum_p n_p l_p of every species against an independent column, the path sampled finely (``oversample``
     points per smallest cell) with each sample in the cell of its nearest generating point, over *all* the cells of
     the container (the Voronoi definition of the gas): a selection that misses a cell, or a deposition that puts gas
     in the wrong cell, shows up here, and the cells met by the samples but absent from the pieces are listed
     (``missing_cells``). Skipped for ``deposition="tophat"`` (spheres, not Voronoi cells: the column agrees only on
     average);
  3. **window**: on one period per skewer with ``velocity_period=True``, the integral of tau over the window against
     its value from the pieces, sum_p C f n_p a_p^-2 l_p lambda (every profile has unit area in frequency): what the
     window holds of the column's absorption, 1 up to the cut at P/2 and O(b / c);
  4. **resolution**: ``optical_depth.resolution_report`` (the pieces narrower than ``tolerance`` b in velocity).

* the ``return_info=True`` option of ``select_cells``, ``simulations.select_cells_near_paths``, ``absorbers.cell_pieces``
  and ``compute_optical_depth``: counts, path lengths, columns, resolution ratios and the time of every stage;
* ``enable_logging(level)``: the package's log messages (``logging.getLogger("skewerjax")``, silent by default) on
  stderr.

Everything here is off by default: the main route never calls it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from skewerjax.core.constants import phys
from skewerjax.core.selection import wrap_positions

logger = logging.getLogger(__name__)


def enable_logging(level=logging.DEBUG, *, stream=None) -> logging.Logger:
    """Show the log messages of SkewerJAX (the logger ``"skewerjax"`` and its children) at ``level`` and above on
    ``stream`` (default stderr). Returns the logger; ``enable_logging(logging.WARNING)`` quietens it again."""
    log = logging.getLogger("skewerjax")
    log.setLevel(level)
    if not any(getattr(h, "_skewerjax", False) for h in log.handlers):
        h = logging.StreamHandler(stream)
        h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s", "%H:%M:%S"))
        h._skewerjax = True
        log.addHandler(h)
    return log


@dataclass
class SkewerCheck:
    """The report of ``check_skewers``: one entry per check (each a dict with ``ok`` and its numbers)."""

    checks: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True if every check that ran passed."""
        return all(c["ok"] for c in self.checks.values() if c.get("ok") is not None)

    def __str__(self) -> str:
        lines = [f"check_skewers: {'OK' if self.ok else 'PROBLEMS FOUND'}"]
        for name, c in self.checks.items():
            flag = "skipped" if c.get("ok") is None else ("ok" if c["ok"] else "FAILED")
            lines.append(f"  {name:10s} {flag:8s} {c.get('summary', '')}")
        return "\n".join(lines)

    __repr__ = __str__


def _fine_samples(points: np.ndarray, spacing: float, max_samples: int):
    """Midpoints and lengths of a fine subdivision of every segment of the polylines ``points`` (N_sk, N, 3): per
    skewer (positions (M, 3), lengths (M,)), sub-intervals <= ``spacing`` (coarser if the total exceeds max_samples)."""
    seg = np.linalg.norm(np.diff(points, axis=1), axis=-1)  # (N_sk, N - 1)
    total = float(np.ceil(seg / spacing).sum())
    if total > max_samples:
        spacing *= total / max_samples
    out = []
    for k in range(points.shape[0]):
        m = np.maximum(np.ceil(seg[k] / spacing).astype(np.int64), 1)
        i = np.repeat(np.arange(seg.shape[1]), m)
        j = np.arange(m.sum()) - np.repeat(np.cumsum(m) - m, m)
        t = (j + 0.5) / m[i]
        pos = points[k, i] + t[:, None] * (points[k, i + 1] - points[k, i])
        out.append((pos, seg[k][i] / m[i]))
    return out, spacing


def check_skewers(
    cells,
    skewer_bundle,
    line_bundle,
    *,
    deposition: str = "voronoi",
    oversample: int = 16,
    max_samples: int = 2_000_000,
    column_tolerance: float = 1e-2,
    window_tolerance: float = 1e-3,
    dv_kms: float | None = None,
    resolution_tolerance: float = 0.25,
    window: bool = True,
) -> SkewerCheck:
    """Independent checks of the pieces and of tau for ``cells`` along ``skewer_bundle`` (module docstring).

    Parameters
    ----------
    cells : ``core.Cells`` holding every species of ``line_bundle`` (all of them: the column check labels the samples
        with the nearest generating point of the whole container).
    skewer_bundle : ``SkewerBundle``; line_bundle : ``LineBundle``.
    deposition : as ``compute_optical_depth`` ("voronoi", "voronoi_crossings" or "tophat").
    oversample : samples per smallest selected cell size along the path (column check); max_samples caps their total.
    column_tolerance : largest relative difference of the column check (the fine sampling has a boundary error of a
        fraction ~1 / oversample of a cell per crossing); window_tolerance : largest |1 - window fraction|.
    dv_kms : pixel of the window check [km/s] (default: a third of the smallest Voigt width of the pieces, at most
        1 km/s); window : False skips the window check (the costliest one: tau over one period per skewer).
    resolution_tolerance : the tolerance of ``resolution_report``.

    Returns
    -------
    ``SkewerCheck``: ``checks`` {"coverage", "column", "window", "resolution"}, each with ``ok`` (None: skipped), a
    ``summary`` line and its arrays; ``info`` the ``return_info`` of ``cell_pieces`` and the time of every check.
    """
    from skewerjax.core.absorbers import cell_pieces
    from skewerjax.core.optical_depth import Pieces, compute_optical_depth, resolution_report

    t_start = time.perf_counter()
    rep = SkewerCheck()
    pieces, parts, info = cell_pieces(
        cells, line_bundle, skewer_bundle=skewer_bundle, deposition=deposition, return_parts=True, return_info=True
    )
    rep.info["cell_pieces"] = info
    pts = np.asarray(skewer_bundle.spatial_points, dtype=np.float64)
    path_len = np.linalg.norm(np.diff(pts, axis=1), axis=-1).sum(-1)
    species = list(line_bundle.species_keys)
    ab0 = parts["absorbers"][species[0]]

    # 1. coverage
    t = time.perf_counter()
    covered = ab0.length.sum(-1)
    rel = np.abs(covered / path_len - 1.0)
    if deposition == "tophat":
        rep.checks["coverage"] = dict(
            ok=None,
            path_length=path_len,
            piece_length=covered,
            summary="skipped: top-hat spheres overlap, their chords do not tile the path",
        )
    else:
        rep.checks["coverage"] = dict(
            ok=bool(np.all(rel < 1e-9)),
            path_length=path_len,
            piece_length=covered,
            rel_error=rel,
            summary=f"sum of piece lengths / path length - 1: max {rel.max():.1e} over {len(rel)} skewers",
        )
    rep.info["time_coverage_s"] = time.perf_counter() - t

    # 2. column against a fine nearest-generator sampling over all the cells
    t = time.perf_counter()
    if deposition == "tophat":
        rep.checks["column"] = dict(ok=None, summary="skipped: top-hat spheres are not the Voronoi cells of the gas")
    else:
        box = cells.box
        gen = wrap_positions(cells.positions, box)
        tree = cKDTree(gen) if box is None else cKDTree(gen, boxsize=box)
        used = parts["cells"]
        h_min = float(np.min(used.size_cMpc))
        samples, spacing = _fine_samples(pts, h_min / max(int(oversample), 1), max_samples)
        sel_idx = info["selection"]["indices"]
        out = {sp: dict(pieces=np.zeros(len(samples)), sampled=np.zeros(len(samples))) for sp in species}
        missing, missing_len = set(), 0.0
        for k, (pos, dl) in enumerate(samples):
            lab = tree.query(wrap_positions(pos, box), k=1, workers=-1)[1]
            c = ab0.cell[k]
            miss = ~np.isin(lab, np.asarray(sel_idx)[c[c >= 0]])  # cells of the samples that hold no piece
            if miss.any():
                missing |= set(np.unique(lab[miss]).tolist())
                missing_len += float(dl[miss].sum())
            for sp in species:
                A = parts["absorbers"][sp]
                out[sp]["pieces"][k] = float((A.n[k] * A.length[k]).sum())
                out[sp]["sampled"][k] = float((np.asarray(cells.species[sp]["number_density"], dtype=np.float64)[lab] * dl).sum())
        worst = 0.0
        for sp in species:
            p, q = out[sp]["pieces"], out[sp]["sampled"]
            out[sp]["rel_error"] = np.abs(p - q) / np.maximum(np.abs(q), 1e-300)
            out[sp]["column_cm2"] = p * phys.Mpc_to_cm  # comoving n_com l [cm^-2]
            worst = max(worst, float(out[sp]["rel_error"].max()))
        ok = worst < column_tolerance and not missing
        rep.checks["column"] = dict(
            ok=ok,
            per_species=out,
            missing_cells=np.array(sorted(missing), dtype=np.int64),
            missing_length_cMpc=missing_len,
            sample_spacing_cMpc=spacing,
            summary=f"pieces vs fine sampling ({spacing:.2g} cMpc): max relative difference {worst:.1e}; "
            f"{len(missing)} cells met by the samples are missing from the pieces "
            f"({missing_len:.2g} cMpc of path)",
        )
    rep.info["time_column_s"] = time.perf_counter() - t

    # 3. the window integral on one period per skewer
    t = time.perf_counter()
    res = resolution_report(pieces, line_bundle, resolution_tolerance)
    if window:
        span = np.asarray(pieces.velocity_span_kms)
        b_min = float(np.min(res["b_min"]))
        dv = min(b_min / 3.0, 1.0) if dv_kms is None else float(dv_kms)
        frac = np.zeros((line_bundle.n_lines, pieces.shape[1]))
        for k in range(pieces.shape[1]):
            n_px = int(np.ceil(span[k] / dv))
            v = (np.arange(n_px) + 0.5) * (span[k] / n_px)
            one = Pieces(
                **{
                    f: (getattr(pieces, f)[k : k + 1] if getattr(pieces, f).ndim < 3 else getattr(pieces, f)[:, k : k + 1])
                    for f in pieces.__dataclass_fields__
                }
            )
            _, ti = compute_optical_depth(one, v, line_bundle, velocity_period=True, resolution_tolerance=None, return_info=True)
            frac[:, k] = ti["window_fraction"][:, 0]
        dev = float(np.max(np.abs(1.0 - frac)))
        rep.checks["window"] = dict(
            ok=dev < window_tolerance,
            window_fraction=frac,
            dv_kms=dv,
            summary=f"integral of tau over one period / its value from the pieces: "
            f"{frac.min():.6f} to {frac.max():.6f} (dv = {dv:.3g} km/s)",
        )
    else:
        rep.checks["window"] = dict(ok=None, summary="skipped (window=False)")
    rep.info["time_window_s"] = time.perf_counter() - t

    # 4. resolution
    rep.checks["resolution"] = dict(
        res,
        ok=bool(res["ok"]),
        summary="column fraction in pieces wider than "
        f"{resolution_tolerance} b: max {np.max(res['column_fraction']):.1e} "
        f"(max W/b = {np.max(res['max_ratio']):.2f})",
    )
    rep.info["time_s"] = time.perf_counter() - t_start
    logger.info("%s", rep)
    return rep
