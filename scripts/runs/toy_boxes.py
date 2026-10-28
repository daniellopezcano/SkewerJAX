"""The toy-box run of notebook 08 (``simulations.run_skewers``) in one process, on the CPU or the GPU.

Several processes can share the run: each claims the next realisation not done (lock files), so the run is resumable
and, for example, one process on the performance cores, one on the efficiency cores and one on the GPU go together:

    taskset -c 0-15  python scripts/runs/toy_boxes.py
    taskset -c 18-31 python scripts/runs/toy_boxes.py
    taskset -c 16-17 python scripts/runs/toy_boxes.py --device gpu

The configuration is ``ToyBoxes`` with the box of the TNG50-4 snapshot of notebooks 05 and 07 and the defaults, and
``SkewerRun()``: set it to notebook 08's to share its run (the notebook itself always recomputes its skewers, emptying
the folder of its configuration first). The results go to ``utils.paths.CACHE_DIR / "toy_boxes"``, keyed by the
configuration. ``--realisations 0 1`` limits the run.
The GPU needs the CUDA libraries of the pip ``nvidia-*`` wheels on ``LD_LIBRARY_PATH``: ``--device gpu`` sets it and
starts the script again.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _gpu_env() -> dict:
    import site

    libs = sorted(str(p) for s in site.getsitepackages() for p in Path(s).glob("nvidia/*/lib"))
    return dict(
        os.environ,
        LD_LIBRARY_PATH=":".join(libs + [os.environ.get("LD_LIBRARY_PATH", "")]),
        JAX_PLATFORMS="cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        SKEWERJAX_TOY_GPU="1",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    ap.add_argument("--realisations", type=int, nargs="*", default=None)
    args = ap.parse_args()
    if args.device == "gpu" and os.environ.get("SKEWERJAX_TOY_GPU") != "1":
        os.execve(sys.executable, [sys.executable, *sys.argv], _gpu_env())
    if args.device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"

    import logging

    logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
    import jax

    jax.config.update("jax_enable_x64", True)
    from skewerjax.simulations import ArepoHDF5Reader, SkewerRun, ToyBoxes, run_skewers
    from skewerjax.utils.paths import CACHE_DIR, DATA_DIR

    box = ArepoHDF5Reader(f"{DATA_DIR}/TNG50-4", 25).header.box_cMpc  # the box of notebooks 05 and 07 [cMpc]
    folder = run_skewers(
        ToyBoxes(box_cMpc=box),
        CACHE_DIR / "toy_boxes",
        SkewerRun(),
        realisations=args.realisations,
        log=lambda s: print(s, flush=True),
    )
    print(f"done: {folder}", flush=True)


if __name__ == "__main__":
    main()
