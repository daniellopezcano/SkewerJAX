"""The named sweeps of ``run_production.py --sweep NAME``: the resolution campaign, four checks that each vary **one**
parameter around a common fiducial run (README.md, "The resolution campaign").

**The fiducial run** (``FIDUCIAL``, the sweep ``fiducial``): TNG50-2 snapshot 25, Lya + Si III 1206, the 128 x 128 lattice
along y, the sampling spacing 0.05 cMpc, 1 km/s pixels, float32, the CPU, the default route (the exact cell crossings, the
integrated profile, the exact frame; with the crossings the spacing is the a(s) table of the frame, set explicitly). It
is computed once: every run of the sweeps that coincides with it (``points/reference``, ``pixels/dv_1kms``,
``skewers/n_0128``, ``simulations/TNG50-2``) and every coarser lattice (``skewers/n_0032``, ``n_0064``) reads its dataset.

* ``points``: the sampling spacing dx = 1, 0.5, 0.1, 0.05, 0.01 cMpc with the sampling points defining the pieces: the
  dual segments of the points (``pieces="segments"``: the cells crossed by the skewer, cut at the midpoints between
  consecutive points), the integrated profile (the package default), the exact frame with its a(s) table on the same
  points; against the fiducial (``reference``: the crossings, no sampling in the pieces) as the converged reference.
* ``pixels``: the pixel width 10, 5, 1, 0.5, 0.1 km/s (nominal: the grid tiles the velocity period exactly, the actual dv
  is recorded).
* ``skewers``: the nested lattices 32^2, 64^2, 128^2, 256^2, 512^2, all in the fiducial's dataset: 256^2 and 512^2 extend it
  (only the skewers it does not hold are computed), the coarser ones are subsets of it.
* ``simulations``: TNG50-2, TNG50-3, TNG50-4 (one box size and one snapshot redshift: the lattice x_i = i L / n puts the
  skewers at the same places in the three boxes).
* ``all``: the four after the fiducial, cheap and foundational first (``ALL_ORDER``): fiducial; simulations; pixels
  (10, 5, 0.5, 0.1 km/s); points (coarse to fine); skewers (256^2, then 512^2, the heaviest).
* ``smoke``: TNG50-4, 2^2 then 4^2 skewers (the second extends the first's dataset) and one 4^2 run of the sampled route,
  8 km/s pixels: a test of the launcher in about a minute.
"""

from __future__ import annotations

from production_runs import RunConfig

FIDUCIAL = RunConfig(
    name="fiducial",
    sim="TNG50-2",
    snap=25,
    n=128,
    axis=2,
    dv_kms=1.0,
    points_cMpc=0.05,
    options={"frame": "exact"},
    precision="float32",
    device="cpu",
)
SAMPLED = {"frame": "exact", "pieces": "segments"}  # the options of the points sweep (the integrated profile: the default)
POINTS_CMPC = (1.0, 0.5, 0.1, 0.05, 0.01)  # coarse to fine
PIXELS_KMS = (10.0, 5.0, 1.0, 0.5, 0.1)
SKEWERS_N = (32, 64, 128, 256, 512)
SIMULATIONS = ("TNG50-2", "TNG50-3", "TNG50-4")
ALL_ORDER = ("fiducial", "simulations", "pixels", "points", "skewers")


def _tag(x: float, unit: str) -> str:
    return f"{x:g}".replace(".", "p") + unit


def fiducial() -> list:
    return [FIDUCIAL]


def points() -> list:
    runs = [FIDUCIAL.replace(name="reference")]
    for dx in POINTS_CMPC:
        runs.append(FIDUCIAL.replace(name=f"dx_{_tag(dx, 'cMpc')}", points_cMpc=dx, options=dict(SAMPLED)))
    return runs


def pixels() -> list:
    return [FIDUCIAL.replace(name=f"dv_{_tag(dv, 'kms')}", dv_kms=dv) for dv in PIXELS_KMS]


def skewers() -> list:
    return [FIDUCIAL.replace(name=f"n_{n:04d}", n=n) for n in SKEWERS_N]


def simulations() -> list:
    return [FIDUCIAL.replace(name=s, sim=s) for s in SIMULATIONS]


def smoke() -> list:
    base = FIDUCIAL.replace(sim="TNG50-4", dv_kms=8.0, tile=8)
    return [base.replace(name=f"n_{n:04d}", n=n) for n in (2, 4)] + [
        base.replace(name="dx_0p5cMpc", n=4, points_cMpc=0.5, options=dict(SAMPLED))
    ]


SWEEPS = {
    "fiducial": (fiducial, "-", "the fiducial run, shared by the four checks"),
    "points": (points, "points_cMpc", "the sampling spacing, the points defining the pieces (dual segments, integrated)"),
    "pixels": (pixels, "dv_kms", "the pixel width"),
    "skewers": (skewers, "n", "the number of skewers (nested lattices of the fiducial's dataset)"),
    "simulations": (simulations, "sim", "the simulation's resolution (TNG50-2, -3, -4)"),
    "smoke": (smoke, "n", "a test of the launcher on TNG50-4 (2^2, 4^2 skewers, one sampled run; 8 km/s)"),
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
