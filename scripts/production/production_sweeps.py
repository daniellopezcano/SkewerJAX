"""The named sweeps of ``run_production.py --sweep NAME``: each a list of runs that vary **one** parameter around a
baseline (README.md, "The four resolution checks").

* ``points``: the spacing of the sampling points, 0.125 to 4 km/s (the benchmark's 0.25 among them), on an 8^2 lattice.
  The default route has no sampling points, so the spacing is varied where it acts: the dual segments with the midpoint
  profile (``segments_<s>``, the route of benchmark block C2) and the a(s) table of the exact frame (``exact_<s>``), each
  against the reference without sampling (``reference``: the default crossings, integrated profile, snapshot frame).
* ``pixels``: the pixel width 1, 2, 4, 8, 16 km/s.
* ``skewers``: the nested lattices 4^2 to 256^2 (the skewers of each lattice are a subset of every finer one).
* ``simulations``: TNG50-2, TNG50-3, TNG50-4 (one box size: the skewers sit at the same places).
* ``all``: the four, in the order that shares the most (``ALL_ORDER``: a run identical to a finished one, e.g. the
  baseline in ``pixels``, ``skewers`` and ``simulations``, is linked, not recomputed).
* ``smoke``: TNG50-4, 2^2 and 4^2 skewers, 8 km/s pixels: a test of the launcher in about a minute.

The baselines: ``BASELINE`` (the benchmark setup at 64^2 skewers: TNG50-2 snapshot 25, Lya + Si III 1206, 1 km/s pixels,
along y, float32, the CPU) for ``pixels``, ``skewers``, ``simulations``; ``points`` keeps every one of its values but the
lattice, 8^2 (the gather route of the sampled pieces selects the cells near every point: C2's lattice, whose cost is
measured; the spacing acts skewer by skewer, so 64 skewers of ~4,000 pixels measure it well).
"""

from __future__ import annotations

from production_runs import RunConfig

BASELINE = RunConfig(name="baseline", sim="TNG50-2", snap=25, n=64, axis=2, dv_kms=1.0, precision="float32", device="cpu")
POINTS_KMS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)  # benchmark block C2's spacings (C2_SPACINGS), the benchmark's 0.25 among them
POINTS_N = 8
PIXELS_KMS = (1.0, 2.0, 4.0, 8.0, 16.0)
SKEWERS_N = (4, 8, 16, 32, 64, 128, 256)
SIMULATIONS = ("TNG50-2", "TNG50-3", "TNG50-4")
ALL_ORDER = ("skewers", "simulations", "pixels", "points")


def _kms(x: float) -> str:
    return f"{x:g}".replace(".", "p") + "kms"


def points() -> list:
    base = BASELINE.replace(n=POINTS_N)
    runs = [base.replace(name="reference")]
    for s in POINTS_KMS:
        runs.append(base.replace(name=f"segments_{_kms(s)}", points_kms=s, options={"pieces": "segments", "profile": "midpoint"}))
    for s in POINTS_KMS:
        runs.append(base.replace(name=f"exact_{_kms(s)}", points_kms=s, options={"frame": "exact"}))
    return runs


def pixels() -> list:
    return [BASELINE.replace(name=f"dv_{_kms(dv)}", dv_kms=dv) for dv in PIXELS_KMS]


def skewers() -> list:
    return [BASELINE.replace(name=f"n_{n:04d}", n=n) for n in SKEWERS_N]


def simulations() -> list:
    return [BASELINE.replace(name=s, sim=s) for s in SIMULATIONS]


def smoke() -> list:
    base = BASELINE.replace(sim="TNG50-4", dv_kms=8.0, tile=8)
    return [base.replace(name=f"n_{n:04d}", n=n) for n in (2, 4)]


SWEEPS = {
    "points": (points, "points_kms", "the spacing of the sampling points, where it acts (dual segments; exact-frame a(s))"),
    "pixels": (pixels, "dv_kms", "the pixel width"),
    "skewers": (skewers, "n", "the number of skewers (nested lattices)"),
    "simulations": (simulations, "sim", "the simulation's resolution (TNG50-2, -3, -4)"),
    "smoke": (smoke, "n", "a test of the launcher on TNG50-4 (2^2, 4^2 skewers, 8 km/s)"),
}


def sweep_runs(name: str, overrides: dict | None = None) -> list:
    """The runs of the sweep ``name`` (``SWEEPS``), every field of ``overrides`` (e.g. {"device": "gpu"}) applied to each
    (the varied field of the sweep cannot be overridden). "all" is not a sweep: ``ALL_ORDER`` lists its sweeps."""
    if name not in SWEEPS:
        raise ValueError(f"unknown sweep {name!r}; the sweeps: {sorted(SWEEPS)} and 'all' ({', '.join(ALL_ORDER)}).")
    make, varied, _ = SWEEPS[name]
    overrides = dict(overrides or {})
    if varied in overrides:
        raise ValueError(f"the sweep {name!r} varies {varied!r}: it cannot be overridden.")
    runs = [r.replace(**overrides) if overrides else r for r in make()]
    names = [r.name for r in runs]
    if len(set(names)) != len(names):
        raise RuntimeError(f"the sweep {name!r} has duplicate run names: {names}")
    return [r.validate() for r in runs]
