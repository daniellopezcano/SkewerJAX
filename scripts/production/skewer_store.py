"""A store of skewers on disk: the optical depth of nested lattices of axis-aligned skewers through a snapshot, kept
for convergence tests and production runs (``STORE.md``; written by ``skewer_extraction``, ``extract_skewers.py`` and
``run_production.py``, read by notebooks 17 and 18). Production code, outside the package; it needs NumPy and h5py only
(not skewerjax), so a store is readable without the package.

**Layout.** One dataset per (simulation, snapshot, axis, options)::

    <root>/<sim>/snap_<NNN>/axis_<a>/<hash>/
        config.json            the identity: the location (sim, path, snapshot, axis) and every option that changes tau
        skewers/batch_*.h5     batches of skewers, each written atomically (a temporary file, fsync, rename)
        runs/run_*.json        one record per invocation of the extraction
        .lock                  held while an extraction writes (one writer per dataset)

``<hash>`` is the first ``HASH_LENGTH`` hex digits of the SHA-256 of the canonical JSON of ``config["options"]`` (the
pixel grid, the lines, the lattice, the gas physics, the pieces, the tau options, the precision): the same options
give the same hash for every simulation, snapshot and axis (the three lines of sight of one configuration share it),
and any new option added to ``options`` changes it automatically. The location is the directory and is checked
against ``config["simulation"]`` when a dataset is opened.

**Nested lattice.** On each transverse axis the skewers sit at x_i = i L / N (offset 0), N a power of two. A skewer is
named by its integer indices (i_a, i_b) on the finest lattice N_ref (``config["options"]["lattice"]["n_ref"]``, a power
of two, default ``N_REF`` = 2^12), so the N x N lattice is the subset of stride N_ref / N, and every coarser lattice is
a subset of every finer one: doubling N reuses every skewer already stored (``lattice_indices``). (a, b) are the two
axes other than the line of sight, in increasing order (``TRANSVERSE``); the skewers of a lattice are ordered a-major,
as ``core.lattice_columns`` and ``summary_stats.on_lattice``.

**Batches.** A batch file holds ``indices`` (m, 2) int32 (on N_ref), ``positions`` (m, 3) float64 [cMpc] (the point of
each skewer at line-of-sight coordinate 0), ``tau/<line_id>`` (m, N_pix) float32, and attributes (the hash, the run,
the git commit, the precision, the device, the resolution check). Nothing else is stored (no flux, no densities). The
index of the stored skewers is rebuilt by scanning the batches (``SkewerDataset.index``): no shared file is rewritten,
and a batch interrupted before its rename leaves only a temporary file, which is ignored and deleted by the next writer.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LAYOUT_VERSION = 1
N_REF = 4096  # 2^12: the finest lattice of a dataset (16.8 M skewers per axis; 12.6 m apart on TNG50's 51.7 cMpc box)
HASH_LENGTH = 10
TRANSVERSE = {1: (1, 2), 2: (0, 2), 3: (0, 1)}  # the transverse axes (a, b) (0-based) of each line of sight 1, 2, 3
_TMP_PREFIX = ".tmp_"


# --------------------------------------------------------------------------- the nested lattice
def is_power_of_two(n) -> bool:
    return isinstance(n, (int, np.integer)) and int(n) > 0 and (int(n) & (int(n) - 1)) == 0


def check_n(n, n_ref: int = N_REF) -> int:
    """``n`` (skewers per transverse axis) as an int, or a ValueError: a power of two, at most ``n_ref``."""
    if isinstance(n, float) and float(n).is_integer():
        n = int(n)
    if not is_power_of_two(n):
        raise ValueError(f"the skewers per axis N must be a power of two (the lattices nest by doubling), got {n!r}.")
    if not is_power_of_two(n_ref):
        raise ValueError(f"N_ref must be a power of two, got {n_ref!r}.")
    if int(n) > int(n_ref):
        raise ValueError(f"N = {n} exceeds the finest lattice of the dataset, N_ref = {n_ref}.")
    return int(n)


def lattice_indices(n: int, n_ref: int = N_REF) -> np.ndarray:
    """(n^2, 2) int64 indices (i_a, i_b) on the N_ref lattice of the n x n lattice (stride N_ref / n), a-major (the order
    of ``core.lattice_columns(n, (1, 1))`` times the stride)."""
    n = check_n(n, n_ref)
    g = np.arange(n, dtype=np.int64) * (int(n_ref) // n)
    return np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)


def lattice_positions(indices, axis: int, box: float, n_ref: int = N_REF) -> np.ndarray:
    """(m, 3) points [cMpc] of the skewers ``indices`` (m, 2) on the N_ref lattice: (i_a, i_b) L / N_ref on the
    transverse axes of ``axis`` (1, 2, 3), 0 along it."""
    idx = np.atleast_2d(np.asarray(indices, dtype=np.int64))
    p = np.zeros((idx.shape[0], 3))
    a, b = TRANSVERSE[int(axis)]
    p[:, a], p[:, b] = idx[:, 0] * (box / n_ref), idx[:, 1] * (box / n_ref)
    return p


def _keys(indices, n_ref: int) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1, 2)
    return idx[:, 0] * int(n_ref) + idx[:, 1]


# --------------------------------------------------------------------------- identity
def canonical_json(x) -> str:
    """The JSON of ``x`` with sorted keys and no spaces (the input of the hash)."""
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_hash(options: dict) -> str:
    """The name of a dataset: the first ``HASH_LENGTH`` hex digits of SHA-256(canonical JSON of ``options``)."""
    return hashlib.sha256(canonical_json(options).encode()).hexdigest()[:HASH_LENGTH]


def dataset_path(root, sim: str, snap: int, axis: int, chash: str) -> Path:
    return Path(root) / str(sim) / f"snap_{int(snap):03d}" / f"axis_{int(axis)}" / chash


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{_TMP_PREFIX}{path.name}.{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- a loaded lattice
@dataclass
class StoredSkewers:
    """The skewers of an n x n lattice of a dataset (``load_skewers``).

    tau : (n_lines, n^2, N_pix) float32, the lines ``line_ids`` on the velocity grid ``v``; indices : (n^2, 2) on the
    N_ref lattice; columns : (n^2, 2) on the n lattice (indices / stride: ``core.lattice_columns(n, (1, 1))``);
    positions : (n^2, 3) [cMpc]; v : (N_pix,) pixel centres [km/s] (the reference line's axis with
    ``line_frame="reference"``), dv [km/s], period [km/s] (= N_pix dv); u = aH [km/s per cMpc]; L : the box [cMpc];
    config : the dataset's config.json."""

    tau: np.ndarray
    indices: np.ndarray
    columns: np.ndarray
    positions: np.ndarray
    v: np.ndarray
    dv: float
    period: float
    u: float
    L: float
    line_ids: tuple
    n_per_axis: int
    sim: str
    snap: int
    axis: int
    hash: str
    config: dict = field(repr=False)

    @property
    def d_perp(self) -> float:
        """The transverse spacing L / n [cMpc]."""
        return self.L / self.n_per_axis


# --------------------------------------------------------------------------- a dataset
class SkewerDataset:
    """One dataset of the store (module docstring): ``SkewerDataset.create(root, config)`` (new, or the existing one
    with the same identity) or ``SkewerDataset.open(root, sim, snap, axis, config_or_hash)``."""

    def __init__(self, path, config: dict):
        self.path = Path(path)
        self.config = config
        o = config["options"]
        self.n_ref = int(o["lattice"]["n_ref"])
        self.line_ids = tuple(o["lines"]["line_ids"])
        self.n_pix = int(o["pixels"]["n_pix"])
        self.hash = config["hash"]
        s = config["simulation"]
        self.sim, self.snap, self.axis = s["name"], int(s["snapshot"]), int(s["axis"])
        self._index = None

    # ---- creation and opening
    @classmethod
    def create(cls, root, config: dict) -> SkewerDataset:
        """The dataset of ``config`` (with "simulation" and "options"; "hash" is added): created with its config.json, or
        the existing one after checking that its identity is the same."""
        config = dict(config)
        config.setdefault("layout_version", LAYOUT_VERSION)
        chash = config_hash(config["options"])
        config["hash"] = chash
        s = config["simulation"]
        path = dataset_path(root, s["name"], s["snapshot"], s["axis"], chash)
        cfg_file = path / "config.json"
        if cfg_file.exists():
            old = json.loads(cfg_file.read_text())
            if canonical_json(old["options"]) != canonical_json(config["options"]):
                raise RuntimeError(
                    f"{cfg_file}: another configuration has the same hash {chash} (a collision): use another root."
                )
            _check_location(old, config, cfg_file)
            return cls(path, old)
        for sub in ("skewers", "runs"):
            (path / sub).mkdir(parents=True, exist_ok=True)
        config.setdefault("created", _now())
        _write_atomic(cfg_file, (json.dumps(config, indent=2, sort_keys=False) + "\n").encode())
        return cls(path, config)

    @classmethod
    def open(cls, root, sim: str, snap: int, axis: int, config_or_hash) -> SkewerDataset:
        """An existing dataset: ``config_or_hash`` its hash (or a unique prefix of it), its config (a dict with
        "options"), or its options."""
        base = Path(root) / str(sim) / f"snap_{int(snap):03d}" / f"axis_{int(axis)}"
        if isinstance(config_or_hash, dict):
            opts = config_or_hash.get("options", config_or_hash)
            chash = config_hash(opts)
        else:
            chash = str(config_or_hash)
        hits = sorted(p for p in base.glob(f"{chash}*") if (p / "config.json").exists()) if base.is_dir() else []
        if not hits:
            have = sorted(p.name for p in base.glob("*") if (p / "config.json").exists()) if base.is_dir() else []
            raise FileNotFoundError(f"no dataset {chash!r} in {base} (there: {have or 'nothing'}).")
        if len(hits) > 1:
            raise ValueError(f"{chash!r} is the prefix of several datasets in {base}: {[p.name for p in hits]}.")
        return cls(hits[0], json.loads((hits[0] / "config.json").read_text()))

    # ---- the index of the stored skewers
    def batch_files(self) -> list:
        return sorted(p for p in (self.path / "skewers").glob("batch_*.h5"))

    def index(self, refresh: bool = False) -> dict:
        """{key: (file, row)} of every stored skewer, key = i_a N_ref + i_b, rebuilt by scanning the batches (a batch
        that cannot be read is reported and ignored: its skewers count as missing; a skewer in two batches keeps the
        first)."""
        if self._index is not None and not refresh:
            return self._index
        import h5py

        out = {}
        for f in self.batch_files():
            try:
                with h5py.File(f, "r") as h:
                    if h.attrs.get("config_hash") != self.hash:
                        raise ValueError(f"hash {h.attrs.get('config_hash')!r}, not {self.hash!r}")
                    keys = _keys(h["indices"][...], self.n_ref)
                    if any(h[f"tau/{lid}"].shape != (keys.size, self.n_pix) for lid in self.line_ids):
                        raise ValueError("tau of the wrong shape")
            except Exception as e:  # noqa: BLE001 (any unreadable batch is skipped, with its reason)
                warnings.warn(f"skewer store: batch {f.name} ignored ({e}); its skewers will be recomputed.", stacklevel=2)
                continue
            for r, k in enumerate(keys.tolist()):
                out.setdefault(k, (f, r))
        self._index = out
        return out

    def stored(self, indices) -> np.ndarray:
        """(m,) bool: which of the skewers ``indices`` (m, 2) are stored."""
        idx = self.index()
        return np.fromiter((k in idx for k in _keys(indices, self.n_ref).tolist()), bool, count=len(np.atleast_2d(indices)))

    def missing(self, n: int) -> np.ndarray:
        """(m, 2) the skewers of the n x n lattice not stored yet."""
        ind = lattice_indices(n, self.n_ref)
        return ind[~self.stored(ind)]

    def complete(self, n: int) -> bool:
        return bool(self.stored(lattice_indices(n, self.n_ref)).all())

    def largest_complete(self) -> int:
        """The largest N whose N x N lattice is complete (0 if none)."""
        best, n = 0, 1
        while n <= self.n_ref and len(self.index()) >= n * n:
            if not self.complete(n):
                break
            best, n = n, 2 * n
        return best

    def complete_ns(self) -> list:
        """Every complete N (1, 2, 4, ..., ``largest_complete``)."""
        top = self.largest_complete()
        return [2**k for k in range(int(np.log2(top)) + 1)] if top else []

    # ---- writing
    @contextlib.contextmanager
    def lock(self):
        """Hold the dataset's lock (one writer at a time; a RuntimeError if another process holds it), and delete the
        temporary files left by an interrupted writer."""
        import fcntl

        f = open(self.path / ".lock", "a+")
        try:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"{self.path} is being written by another process (its .lock is held).") from None
            for sub in ("skewers", "runs"):
                for t in (self.path / sub).glob(f"{_TMP_PREFIX}*"):
                    t.unlink()
            yield self
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()

    def write_batch(self, indices, tau, *, name: str, attrs: dict | None = None) -> Path:
        """Write one batch atomically: ``indices`` (m, 2) on N_ref, ``tau`` (n_lines, m, N_pix) (stored float32), as
        ``skewers/batch_<name>.h5``; ``attrs`` (JSON-able values) are stored as attributes. Returns the path."""
        import h5py

        idx = np.asarray(indices, dtype=np.int64).reshape(-1, 2)
        tau = np.asarray(tau)
        if tau.shape != (len(self.line_ids), idx.shape[0], self.n_pix):
            raise ValueError(
                f"tau must be (n_lines, m, N_pix) = {(len(self.line_ids), idx.shape[0], self.n_pix)}, got {tau.shape}."
            )
        if idx.size and (idx.min() < 0 or idx.max() >= self.n_ref):
            raise ValueError(f"indices outside the N_ref = {self.n_ref} lattice.")
        final = self.path / "skewers" / f"batch_{name}.h5"
        if final.exists():
            raise FileExistsError(final)
        tmp = final.with_name(f"{_TMP_PREFIX}{final.name}")
        L = float(self.config["snapshot"]["box_cMpc"])
        with h5py.File(tmp, "w") as h:
            h.create_dataset("indices", data=idx.astype(np.int32))
            h.create_dataset("positions", data=lattice_positions(idx, self.axis, L, self.n_ref))
            g = h.create_group("tau")
            for j, lid in enumerate(self.line_ids):
                g.create_dataset(lid, data=tau[j].astype(np.float32), chunks=(min(64, max(idx.shape[0], 1)), self.n_pix))
            h.attrs["config_hash"] = self.hash
            h.attrs["layout_version"] = LAYOUT_VERSION
            h.attrs["n_ref"] = self.n_ref
            h.attrs["axis"] = self.axis
            h.attrs["line_ids"] = json.dumps(list(self.line_ids))
            h.attrs["written"] = _now()
            for k, val in (attrs or {}).items():
                h.attrs[k] = val if isinstance(val, (str, int, float, bool)) else json.dumps(val)
        with open(tmp, "rb+") as f:  # on disk before the rename: a crash leaves the old state or the whole batch
            os.fsync(f.fileno())
        os.replace(tmp, final)
        if self._index is not None:
            for r, k in enumerate(_keys(idx, self.n_ref).tolist()):
                self._index.setdefault(k, (final, r))
        return final

    def write_run(self, record: dict) -> Path:
        """Write (or rewrite) the record of one run, ``runs/run_<record["run_id"]>.json``, atomically."""
        p = self.path / "runs" / f"run_{record['run_id']}.json"
        _write_atomic(p, (json.dumps(record, indent=1, default=_json_default) + "\n").encode())
        return p

    def sources(self, n: int) -> list:
        """Where the n x n lattice comes from, level by level of the nested lattices: one row per N = 1, 2, 4, ... n,
        dict(n (the level), skewers (those of the n x n lattice first on the N x N lattice: N^2 - (N/2)^2, 1 for N = 1),
        stored (how many of them are stored), files (the batch files that hold them, sorted)). Loading the n x n lattice
        reads every level: its skewers are those of every coarser lattice plus the new ones of each doubling."""
        n = check_n(n, self.n_ref)
        ind = lattice_indices(n, self.n_ref)
        idx = self.index(refresh=True)
        keys = _keys(ind, self.n_ref).tolist()
        stride = self.n_ref // n
        # the coarsest lattice N holding a skewer: n / gcd(c_a, c_b, n), (c_a, c_b) its indices on the n lattice
        # (gcd(0, n) = n: the skewer (0, 0) is on N = 1)
        level = n // np.gcd(np.gcd(ind[:, 0] // stride, ind[:, 1] // stride), n)
        rows = []
        m = 1
        while m <= n:
            sel = np.flatnonzero(level == m)
            files = sorted({idx[keys[i]][0].name for i in sel if keys[i] in idx})
            rows.append(dict(n=m, skewers=int(sel.size), stored=int(sum(keys[i] in idx for i in sel)), files=files))
            m *= 2
        return rows

    def runs(self) -> list:
        """The records of every run, oldest first."""
        return [json.loads(p.read_text()) for p in sorted((self.path / "runs").glob("run_*.json"))]

    # ---- reading
    def load(self, n: int) -> StoredSkewers:
        """The n x n lattice (``StoredSkewers``); a ValueError naming what is stored if it is not complete."""
        import h5py

        n = check_n(n, self.n_ref)
        ind = lattice_indices(n, self.n_ref)
        idx = self.index(refresh=True)
        keys = _keys(ind, self.n_ref).tolist()
        miss = sum(k not in idx for k in keys)
        if miss:
            raise ValueError(
                f"{self.path}: the {n} x {n} lattice is not complete ({miss} of {n * n} skewers missing; the largest complete "
                f"N is {self.largest_complete()}). Run scripts/production/extract_skewers.py with --n-per-axis {n} to fill it."
            )
        tau = np.empty((len(self.line_ids), n * n, self.n_pix), np.float32)
        by_file = {}
        for i, k in enumerate(keys):
            f, r = idx[k]
            by_file.setdefault(f, []).append((r, i))
        for f, pairs in by_file.items():
            pairs.sort()
            rows = np.array([r for r, _ in pairs])
            dest = np.array([i for _, i in pairs])
            with h5py.File(f, "r") as h:
                for j, lid in enumerate(self.line_ids):
                    d = h[f"tau/{lid}"]
                    tau[j, dest] = d[...][rows] if rows.size > d.shape[0] // 4 else d[rows]
        p = self.config["options"]["pixels"]
        snap = self.config["snapshot"]
        n_pix, dv = int(p["n_pix"]), float(p["dv_kms"])
        L = float(snap["box_cMpc"])
        return StoredSkewers(
            tau=tau,
            indices=ind,
            columns=ind // (self.n_ref // n),
            positions=lattice_positions(ind, self.axis, L, self.n_ref),
            v=(np.arange(n_pix) + 0.5) * dv,
            dv=dv,
            period=float(p["period_kms"]),
            u=float(snap["u_kms_per_cMpc"]),
            L=L,
            line_ids=self.line_ids,
            n_per_axis=n,
            sim=self.sim,
            snap=self.snap,
            axis=self.axis,
            hash=self.hash,
            config=self.config,
        )

    def summary(self) -> dict:
        """One row of ``list_datasets``."""
        o = self.config["options"]
        return dict(
            sim=self.sim,
            snap=self.snap,
            axis=self.axis,
            hash=self.hash,
            preset=self.config.get("preset"),
            n_max=self.largest_complete(),
            n_stored=len(self.index(refresh=True)),
            spacing_kms=o["pixels"].get("spacing_kms"),
            n_pix=int(o["pixels"]["n_pix"]),
            line_ids=",".join(self.line_ids),
            precision=o.get("precision"),
            path=str(self.path),
        )


def _check_location(old: dict, new: dict, cfg_file) -> None:
    """The location of an existing dataset against a new request: sim name, snapshot and axis must match (they are the
    directory); a different simulation path is accepted, with a warning, only if the snapshot's header is the same (the
    data moved), otherwise a ValueError (another simulation under the same name)."""
    a, b = old["simulation"], new["simulation"]
    for k in ("name", "snapshot", "axis"):
        if a[k] != b[k]:
            raise ValueError(f"{cfg_file}: {k} {a[k]!r} in config.json, {b[k]!r} requested.")
    if a.get("path") != b.get("path"):
        same = all(old["snapshot"].get(k) == new.get("snapshot", {}).get(k) for k in ("n_gas", "box_cMpc", "redshift"))
        if not same:
            raise ValueError(
                f"{cfg_file}: simulation {a['name']!r} was {a.get('path')!r}, now {b.get('path')!r} with another header: use another --sim name."
            )
        warnings.warn(
            f"{cfg_file}: simulation path {a.get('path')!r} -> {b.get('path')!r} (the same header: the data moved).", stacklevel=3
        )


def _json_default(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"not JSON serialisable: {type(x)}")


# --------------------------------------------------------------------------- the public reading API
def load_skewers(root, sim: str, snap: int, axis: int, config_or_hash, n_per_axis: int) -> StoredSkewers:
    """The n x n lattice of skewers of one dataset (``StoredSkewers``: tau (n_lines, n^2, N_pix), the positions, the
    velocity grid, the period, u, L, the line ids and the config). ``config_or_hash``: the hash (or a unique prefix), a
    config or its options. A ValueError if the lattice is not complete."""
    return SkewerDataset.open(root, sim, snap, axis, config_or_hash).load(n_per_axis)


def list_datasets(root) -> list:
    """Every dataset under ``root``: a list of rows (dicts: sim, snap, axis, hash, preset, n_max (the largest complete
    N), n_stored, spacing_kms, n_pix, line_ids, precision, path), sorted. ``format_datasets`` prints them."""
    rows = []
    for cfg in sorted(Path(root).glob("*/snap_*/axis_*/*/config.json")):
        try:
            rows.append(SkewerDataset(cfg.parent, json.loads(cfg.read_text())).summary())
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"{cfg}: not a dataset ({e}).", stacklevel=2)
    return sorted(rows, key=lambda r: (r["sim"], r["snap"], r["hash"], r["axis"]))


def select_datasets(root, sim: str, snap: int, config_or_hash=None, axes=(1, 2, 3)) -> dict:
    """The datasets of one configuration for an ensemble of lines of sight (notebook 17): ``config_or_hash`` (a hash or a
    unique prefix, a config, or None: the configuration whose largest lattice complete on every one of its axes is the
    largest, then the one with the most axes) of the simulation ``sim``, snapshot ``snap``, on the ``axes`` it has.
    Returns dict(hash, axes (those of ``axes`` with the dataset), n_values (the N complete on every one of them,
    increasing), n_max, rows (of ``list_datasets``)); a FileNotFoundError if there is none."""
    rows = [r for r in list_datasets(root) if r["sim"] == sim and r["snap"] == int(snap) and r["axis"] in tuple(axes)]
    if isinstance(config_or_hash, dict):
        config_or_hash = config_hash(config_or_hash.get("options", config_or_hash))
    if config_or_hash is not None:
        rows = [r for r in rows if r["hash"].startswith(str(config_or_hash))]
    if not rows:
        raise FileNotFoundError(f"no dataset of {sim} snapshot {snap} ({config_or_hash or 'any configuration'}) in {root}.")
    by_hash = {}
    for r in rows:
        by_hash.setdefault(r["hash"], []).append(r)
    if len({h for h in by_hash}) > 1 and config_or_hash is not None:
        raise ValueError(f"{config_or_hash!r} is the prefix of several configurations: {sorted(by_hash)}.")
    best = max(by_hash, key=lambda h: (min(r["n_max"] for r in by_hash[h]), len(by_hash[h])))
    sel = sorted(by_hash[best], key=lambda r: r["axis"])
    top = min(r["n_max"] for r in sel)
    ns = [2**k for k in range(int(np.log2(top)) + 1)] if top else []
    return dict(hash=best, axes=tuple(r["axis"] for r in sel), n_values=ns, n_max=top, rows=sel)


def load_run(run_dir, n_per_axis: int | None = None) -> StoredSkewers:
    """The skewers of a run folder of ``run_production.py`` (``<out>/<sweep>/<run>/``): its ``run.json`` names the dataset
    of its store (``run["dataset"]``: root, sim, snap, axis, hash, n_per_axis); ``n_per_axis`` default the run's lattice
    (any coarser power of two is a subset of it). A ValueError if the run is not finished."""
    run_dir = Path(run_dir).expanduser()
    rec = json.loads((run_dir / "run.json").read_text())
    if rec.get("status") != "done" or "dataset" not in rec:
        raise ValueError(f"{run_dir}: the run is not finished (status {rec.get('status')!r}).")
    d = rec["dataset"]
    n = int(d["n_per_axis"] if n_per_axis is None else n_per_axis)
    return load_skewers(run_dir / d["root"], d["sim"], d["snap"], d["axis"], d["hash"], n)


def list_runs(sweep_dir) -> list:
    """The runs of a sweep folder (its ``index.json``, else the ``run.json`` of every sub-folder): rows of name, status,
    path, the run's configuration and dataset."""
    sweep_dir = Path(sweep_dir).expanduser()
    rows = []
    for p in sorted(sweep_dir.glob("*/run.json")):
        rec = json.loads(p.read_text())
        rows.append(
            dict(
                name=rec.get("name", p.parent.name),
                status=rec.get("status"),
                path=str(p.parent),
                config=rec.get("config"),
                dataset=rec.get("dataset"),
                shared_from=rec.get("shared_from"),
            )
        )
    return rows


def format_datasets(rows) -> str:
    """The rows of ``list_datasets`` as a text table."""
    cols = ("sim", "snap", "axis", "hash", "preset", "spacing_kms", "n_pix", "line_ids", "precision", "n_stored", "n_max")
    if not rows:
        return "(no dataset)"
    txt = [[str(r.get(c)) for c in cols] for r in rows]
    w = [max(len(c), *(len(t[i]) for t in txt)) for i, c in enumerate(cols)]
    line = lambda xs: "  ".join(x.ljust(w[i]) for i, x in enumerate(xs))
    return "\n".join([line(cols), line(["-" * x for x in w]), *(line(t) for t in txt)])
