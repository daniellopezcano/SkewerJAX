from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import glob
import logging
import numpy as np
import h5py


@dataclass
class SnapshotInfo:
    snapnum: int
    snapdir: Path
    files: list[Path]
    header: dict
    parameters: dict
    config: dict
    cosmo: dict


class TNGSimulation:
    """
    Minimal helper to inspect an Arepo/Gadget-style HDF5 simulation directory:
      base/
        snapdir_000/
          snap_000.0.hdf5 ...
        snapdir_001/
          snap_001.0.hdf5 ...
        ...

    It discovers available snapshots, reads per-snapshot metadata, builds a small cosmology dict,
    and checks metadata consistency across snapshots.
    """

    def __init__(self, base: str | Path, logger: logging.Logger | None = None):

        self.log = logger or logging.getLogger(self.__class__.__name__)

        self.base = Path(base).expanduser().resolve()
        self._validate_base()        

        self.snapnums: list[int] = []
        self._discover_snapshots()

        self.snapshots: dict[int, SnapshotInfo] = {}
        self.cosmo: dict = {}
        self._read_all_snapshots()

        self._check_consistency()

    # -------------------------
    # Public convenience
    # -------------------------
    def get(self, snapnum: int) -> SnapshotInfo:
        return self.snapshots[snapnum]

    def summary(self) -> dict:
        """Return a compact summary dict."""
        return {
            "base": str(self.base),
            "n_snapshots": len(self.snapnums),
            "snapnums": list(self.snapnums),
            "cosmo": dict(self.cosmo),
        }

    # -------------------------
    # Internals
    # -------------------------
    def _validate_base(self) -> None:
        if not self.base.exists():
            raise FileNotFoundError(f"Base path does not exist: {self.base}")
        if not self.base.is_dir():
            raise NotADirectoryError(f"Base path is not a directory: {self.base}")
        self.log.info("Simulation base: %s", self.base)

    def _discover_snapshots(self) -> None:
        snapdirs = sorted(self.base.glob("snapdir_*"))
        snapnums = []
        for sd in snapdirs:
            try:
                n = int(sd.name.split("_")[-1])
                snapnums.append(n)
            except Exception:
                continue

        snapnums = sorted(set(snapnums))
        self.snapnums = snapnums

        if not snapnums:
            raise FileNotFoundError(f"No snapdir_XXX directories found under {self.base}")

        self.log.info("Discovered %d snapshot directories.", len(snapnums))
        self.log.info("Snapshot numbers: %s", snapnums)

    @staticmethod
    def _attrs_to_dict(h5obj) -> dict:
        out = {}
        for k in h5obj.attrs.keys():
            v = h5obj.attrs[k]
            if isinstance(v, np.ndarray) and v.shape == ():
                v = v.item()
            out[k] = v
        return out

    @staticmethod
    def _get_any(d: dict, candidates: list[str], default=None):
        for c in candidates:
            if c in d:
                return d[c]
        low = {k.lower(): k for k in d.keys()}
        for c in candidates:
            if c.lower() in low:
                return d[low[c.lower()]]
        return default

    def _list_snapshot_files(self, snapnum: int) -> tuple[Path, list[Path]]:
        snapdir = self.base / f"snapdir_{snapnum:03d}"
        if not snapdir.exists():
            raise FileNotFoundError(f"Missing directory: {snapdir}")

        files = sorted(Path(p).resolve() for p in glob.glob(str(snapdir / f"snap_{snapnum:03d}*.hdf5")))
        if len(files) == 0:
            raise FileNotFoundError(f"No snapshot files found in {snapdir} matching snap_{snapnum:03d}*.hdf5")
        return snapdir, files

    def _parse_cosmo(self, header: dict, params: dict) -> dict:
        # TNG/AREPO common keys
        cosmo = {}
        cosmo["BoxSize"] = self._get_any(header, ["BoxSize"])
        cosmo["Redshift"] = self._get_any(header, ["Redshift"])
        cosmo["ScaleFactor"] = self._get_any(header, ["Time"])  # typically a

        cosmo["h"] = self._get_any(header, ["HubbleParam", "HubbleParameter"])
        cosmo["Omega_m"] = self._get_any(header, ["Omega0", "OmegaM", "Omega_m"])
        cosmo["Omega_L"] = self._get_any(header, ["OmegaLambda", "OmegaL", "Omega_Lambda"])
        cosmo["Omega_b"] = self._get_any(header, ["OmegaBaryon", "OmegaBaryon0", "OmegaB"])
        cosmo["Omega_k"] = self._get_any(header, ["OmegaK", "OmegaCurvature", "Omega_k"])

        # Sometimes in Parameters (not always in snapshots)
        cosmo["sigma8"] = self._get_any(params, ["Sigma8", "sigma8"], default=self._get_any(header, ["Sigma8", "sigma8"]))
        cosmo["ns"] = self._get_any(params, ["PrimordialIndex", "ns", "ScalarSpectralIndex"], default=self._get_any(header, ["ns"]))
        cosmo["As"] = self._get_any(params, ["PrimordialAmplitude", "As", "ScalarAmplitude"], default=self._get_any(header, ["As"]))

        # Useful extra header fields
        cosmo["NumFilesPerSnapshot"] = self._get_any(header, ["NumFilesPerSnapshot"])
        cosmo["MassTable"] = self._get_any(header, ["MassTable"])
        cosmo["NumPart_Total"] = self._get_any(header, ["NumPart_Total"])
        cosmo["NumPart_Total_HighWord"] = self._get_any(header, ["NumPart_Total_HighWord"])
        return cosmo

    def _read_one_snapshot(self, snapnum: int) -> SnapshotInfo:
        snapdir, files = self._list_snapshot_files(snapnum)
        fn0 = files[0]
        self.log.info("Reading snapshot %03d (first file: %s)", snapnum, fn0.name)

        with h5py.File(fn0, "r") as f:
            if "Header" not in f:
                raise KeyError(f"Missing 'Header' group in {fn0}")
            header = self._attrs_to_dict(f["Header"])
            params = self._attrs_to_dict(f["Parameters"]) if "Parameters" in f else {}
            config = self._attrs_to_dict(f["Config"]) if "Config" in f else {}

        cosmo = self._parse_cosmo(header, params)

        # Minimal logging summary
        z = cosmo.get("Redshift", None)
        a = cosmo.get("ScaleFactor", None)
        box = cosmo.get("BoxSize", None)
        self.log.info("  z=%s  a=%s  BoxSize=%s  nfiles=%d", z, a, box, len(files))

        # Mention missing common keys (normal sometimes)
        missing = [k for k in ["h", "Omega_m", "Omega_L", "Omega_b", "sigma8", "ns", "As"] if cosmo.get(k, None) is None]
        if missing:
            self.log.info("  Missing in metadata (may be normal): %s", missing)

        return SnapshotInfo(
            snapnum=snapnum,
            snapdir=snapdir,
            files=files,
            header=header,
            parameters=params,
            config=config,
            cosmo=cosmo,
        )

    def _read_all_snapshots(self) -> None:
        for n in self.snapnums:
            info = self._read_one_snapshot(n)
            self.snapshots[n] = info

    @staticmethod
    def _nearly_equal(a, b, rtol=1e-8, atol=0.0) -> bool:
        try:
            return bool(np.isclose(a, b, rtol=rtol, atol=atol))
        except Exception:
            return False

    def _check_consistency(self) -> None:
        """
        Check that key metadata fields are consistent across snapshots.
        We expect: h, Omega_m, Omega_L, Omega_b, BoxSize, NumFilesPerSnapshot (often), MassTable (often).
        Redshift/ScaleFactor should differ across snapshots, so they are excluded from the consistency set.
        """
        if not self.snapnums:
            return

        keyset = ["h", "Omega_m", "Omega_L", "Omega_b", "BoxSize", "NumFilesPerSnapshot"]
        ref = self.snapshots[self.snapnums[0]].cosmo

        inconsistent = []
        for n in self.snapnums[1:]:
            c = self.snapshots[n].cosmo
            for k in keyset:
                v0 = ref.get(k, None)
                v1 = c.get(k, None)
                # Only compare if both are present
                if v0 is None or v1 is None:
                    continue
                ok = self._nearly_equal(v0, v1) if isinstance(v0, (int, float, np.number)) else (v0 == v1)
                if not ok:
                    inconsistent.append((k, self.snapnums[0], v0, n, v1))

        if inconsistent:
            for k, n0, v0, n1, v1 in inconsistent[:20]:
                self.log.info("Metadata mismatch for %s: snap%03d=%s vs snap%03d=%s", k, n0, v0, n1, v1)
            raise ValueError(f"Found {len(inconsistent)} metadata inconsistencies across snapshots (see logs).")

        # Build "common cosmology" from first snapshot, keeping only stable keys
        common = {}
        for k, v in ref.items():
            if k in ["Redshift", "ScaleFactor"]:
                continue
            common[k] = v
        self.cosmo = common

        self.log.info("Metadata appears self-consistent across %d snapshots.", len(self.snapnums))
