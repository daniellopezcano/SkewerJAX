"""The cost of a production run before it starts: its disk size, wall time and peak memory (``run_production.py
--dry-run``; README.md, "Estimates"). Nothing here runs SkewerJAX's engine: the times come from the measured benchmark
records of ``benchmarks/results`` (``skewerjax.utils.paths.BENCHMARK_RECORDS_DIR``), the memory from the pass's memory
model of ``skewer_extraction`` (notebook 16), the disk from the store's format.

**Disk** (exact up to the HDF5 overhead): every skewer stores ``tau`` of every line as float32 over N_pix pixels, its
indices (2 int32) and its position (3 float64), so n^2 (4 n_lines N_pix + 32) bytes, plus ~0.1 MB per batch file
(HDF5 headers, attributes, chunk index) and the JSON records.

**Wall time** of the default streaming route (``pieces`` "crossings"; "chords" for the "matched" preset), from the
scaling records (cold page cache, float32, the CPU as two JAX devices; notebook 14):

* TNG50-2 at 1 km/s: the measured totals of the entry ("SkewerJAX default, CPU" / "GPU", "SkewerJAX matched, CPU")
  at 4^2, 16^2, 32^2, 64^2, 128^2 and 256^2 skewers, interpolated linearly in log(number of skewers) (beyond 256^2:
  extrapolated linearly in the number of skewers from the last two points; below 4^2: the 4^2 value);
* the pixel width: only the tau stage depends on it; its ratio to 1 km/s at 128^2 (block C1: 1, 2, 4, 8, 16 km/s) is
  interpolated in log(dv) and applied to the tau stage of the interpolated total (the tau stage's share of the total at
  the nearest record);
* another simulation: the TNG50-2 estimate times the ratio of that simulation's measured total at 128^2 to TNG50-2's
  (TNG50-3, TNG50-4 at 128^2; a simulation without records: the ratio of its number of cells, an assumption);
* plus the compilation (the record's ``jit_s`` nearest in skewers: each run is a fresh process), plus one extra pass per
  extra pass over the snapshot when the memory model splits the run (``skewer_extraction.pass_size``);
* float64: the tau stage doubled (an assumption, not measured).

The sampled route (``pieces="segments"``, the midpoint profile): the selection and gas physics taken as the pass of the
streaming route at the same lattice (the gather route reads the snapshot once, as the pass; not measured separately),
plus twice the tau time per skewer of block C2 (``C2.json``, TNG50-2, 8^2 skewers, float64; the pieces' deposition is not
in C2's timing and is assumed to cost as much again), interpolated in log(points_kms). A rough estimate.

**Peak memory**: ``skewer_extraction.pass_memory`` (5.7 GB + 92 B per cell within reach + the pass's tau), the model that
splits a run into passes; against notebook 16's measurements it is conservative (TNG50-2: 37 GB modelled / 29.8 GB measured
at 128^2, 68 / 65.7 GB at 256^2).
"""

from __future__ import annotations

import json
import math
from functools import cache
from pathlib import Path

HDF5_BATCH_OVERHEAD = 1e5  # bytes per batch file (headers, attributes, chunk index), measured on TNG50-4 batches ~ 3e4
ENTRIES = {
    ("default", "cpu"): "SkewerJAX default, CPU",
    ("default", "gpu"): "SkewerJAX default, GPU",
    ("matched", "cpu"): "SkewerJAX matched, CPU",
    ("matched", "gpu"): "SkewerJAX matched, CPU",
}
NBINS_1KMS = {"TNG50-2": 3961, "TNG50-3": 3961, "TNG50-4": 3961}  # pixels of 1 km/s over one box at z = 3 (the records)
N_GAS = {"TNG50-2": 1205098922, "TNG50-3": 152305397, "TNG50-4": 19250411}  # the records' snapshots (snapshot 25)


def _records_dir(records_dir=None) -> Path:
    if records_dir is not None:
        return Path(records_dir)
    from skewerjax.utils.paths import BENCHMARK_RECORDS_DIR

    return Path(BENCHMARK_RECORDS_DIR)


@cache
def load_records(records_dir=None) -> dict:
    """The scaling records ({id: record}) and C2 of ``benchmarks/results/scaling`` (empty if absent)."""
    folder = _records_dir(records_dir) / "scaling"
    out = {}
    for p in sorted(folder.glob("*.json")):
        try:
            r = json.loads(p.read_text())
            out[r.get("id", p.stem)] = r
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _interp(x, xs, ys, log_x=True):
    """Linear interpolation of ``ys`` at ``x`` over increasing ``xs`` (in log x by default); constant below, linear in x
    (not log) from the last two points above."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        if len(xs) < 2:
            return ys[-1]
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return ys[-1] + slope * (x - xs[-1])
    f = (lambda v: math.log(v)) if log_x else (lambda v: v)
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (f(x) - f(xs[i - 1])) / (f(xs[i]) - f(xs[i - 1]))
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def _sim_points(records: dict, sim: str, entry: str) -> list:
    """[(n_skewers, total_s, tau_s, jit_s)] of the 1 km/s grid of ``sim`` for ``entry``, by number of skewers."""
    pts = []
    for r in records.values():
        c = r.get("config", {})
        if c.get("sim") != sim or "grids" not in r:
            continue
        g = next((k for k, v in r["grids"].items() if abs(float(v["dv_kms"]) - 1.0) < 0.01), None)
        e = r.get("entries", {}).get(f"{entry}|{g}") if g is not None else None
        if e is None:
            continue
        pts.append((int(c["n_skewers"]), float(e["total"]), float(e["stages"]["tau"]), float(r.get("jit_s", 0.0))))
    return sorted(pts)


def _dv_ratio(records: dict, entry: str, dv: float) -> float:
    """The tau stage at ``dv`` over the tau stage at 1 km/s, from the 128^2 TNG50-2 record of block C1."""
    r = records.get("TNG50-2_n128_dv1-2-4-8-16")
    if r is None:
        return 1.0
    pts = []
    for g, v in r["grids"].items():
        e = r["entries"].get(f"{entry}|{g}")
        if e is not None:
            pts.append((float(v["dv_kms"]), float(e["stages"]["tau"])))
    pts.sort()
    if len(pts) < 2:
        return 1.0
    t1 = _interp(1.0, [p[0] for p in pts], [p[1] for p in pts])
    return _interp(max(dv, pts[0][0]), [p[0] for p in pts], [p[1] for p in pts]) / t1


def streaming_time(
    sim: str, n_skewers: int, dv_kms: float, *, mode="default", device="cpu", precision="float32", records=None
) -> dict:
    """The estimated wall time [s] of one pass of the streaming route (module docstring): dict(total_s, compute_s,
    tau_s, pass_s, jit_s, basis)."""
    records = load_records() if records is None else records
    entry = ENTRIES.get((mode, device), ENTRIES[("default", "cpu")])
    base = _sim_points(records, "TNG50-2", entry)
    if not base:
        return dict(
            total_s=float("nan"),
            compute_s=float("nan"),
            tau_s=float("nan"),
            pass_s=float("nan"),
            jit_s=30.0,
            basis="no benchmark records found",
        )
    xs = [p[0] for p in base]
    total = _interp(n_skewers, xs, [p[1] for p in base])
    tau1 = _interp(n_skewers, xs, [p[2] for p in base])
    jit = base[min(range(len(xs)), key=lambda i: abs(math.log(xs[i]) - math.log(max(n_skewers, 1))))][3]
    tau = tau1 * _dv_ratio(records, entry, dv_kms)
    basis = f"TNG50-2 '{entry}' records at {xs[0]}-{xs[-1]} skewers (log-interpolated), the tau stage scaled to {dv_kms:g} km/s"
    scale = 1.0
    if sim != "TNG50-2":
        own = {p[0]: p[1] for p in _sim_points(records, sim, entry)}
        ref = {p[0]: p[1] for p in base}
        common = sorted(set(own) & set(ref))
        if common:
            n0 = common[-1]
            scale = own[n0] / ref[n0]
            basis += f"; x {scale:.3f} = {sim} / TNG50-2 at {n0} skewers"
        else:
            scale = N_GAS.get(sim, N_GAS["TNG50-2"]) / N_GAS["TNG50-2"]
            basis += f"; x {scale:.3f} = the ratio of cells (no {sim} record: an assumption)"
    pass_s = max(total - tau1, 0.0)
    if precision == "float64":
        tau *= 2.0
        basis += "; tau doubled for float64 (not measured)"
    compute = scale * (pass_s + tau)
    return dict(total_s=compute + jit, compute_s=compute, tau_s=scale * tau, pass_s=scale * pass_s, jit_s=jit, basis=basis)


def sampled_time(n_skewers: int, points_kms: float, *, precision="float32", records=None) -> dict:
    """The estimated wall time [s] of the sampled route (module docstring): dict(total_s, ..., basis)."""
    records = load_records() if records is None else records
    st = streaming_time("TNG50-2", n_skewers, 1.0, records=records)
    c2 = records.get("C2", {}).get("midpoint", {})
    pts = sorted((float(k), float(v["tau_s_per_skewer"])) for k, v in c2.items())
    if pts:
        per = _interp(points_kms, [p[0] for p in pts], [p[1] for p in pts])
    else:
        per = 0.03
    tau = 2.0 * per * n_skewers  # C2's tau time per skewer, doubled for the deposition (not in C2's timing)
    compute = st["pass_s"] + tau
    basis = (
        f"selection ~ the TNG50-2 pass at {n_skewers} skewers ({st['pass_s']:.0f} s, not measured for this route) + "
        f"2 x C2's tau time per skewer at {points_kms:g} km/s ({per * 1e3:.1f} ms, float64): rough"
    )
    return dict(total_s=compute + st["jit_s"], compute_s=compute, tau_s=tau, pass_s=st["pass_s"], jit_s=st["jit_s"], basis=basis)


def disk_bytes(n_skewers: int, n_lines: int, n_pix: int, tile: int) -> float:
    """The bytes a run writes: tau float32, the indices and positions of every skewer, the batch overhead, the JSON."""
    batches = max(1, math.ceil(n_skewers / max(int(tile), 1)))
    return n_skewers * (4.0 * n_lines * n_pix + 32.0) + batches * HDF5_BATCH_OVERHEAD + 2e5


def n_pix_estimate(sim: str, dv_kms: float | None, n_pix: int | None) -> int:
    """The pixels of a run without reading its snapshot (the records' grid at 1 km/s scaled by dv; the exact value comes
    from ``skewer_extraction.dataset_options`` when the header is read)."""
    if n_pix:
        return int(n_pix)
    return max(2, int(NBINS_1KMS.get(sim, 3961) / float(dv_kms or 1.0)))
