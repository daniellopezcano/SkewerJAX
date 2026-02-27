from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import glob
import logging
import numpy as np
import jax.numpy as jnp
import h5py

from scipy.spatial import cKDTree

from skewerjax.constants.lines import get_mass_amu
from skewerjax.constants.physical_constants import phys
from skewerjax.simulations.cloudy_tables import convert_cloudy
from importlib import resources

def get_default_cloudy_table_dir(subdir="ion_out_photo_atten"):
    base = resources.files("skewerjax.simulations.cloudy_tables")
    return str(base.joinpath(subdir))

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
            if isinstance(v, jnp.ndarray) and v.shape == ():
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
            return bool(jnp.isclose(a, b, rtol=rtol, atol=atol))
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
                ok = self._nearly_equal(v0, v1) if isinstance(v0, (int, float, jnp.number)) else (v0 == v1)
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


    def load_snapdir_hdf5(
        self,
        snapnum: int,
        part_group: str = "PartType0",
        datasets: list[str] | None = None,
        file_indices: list[int] | None = None,
        dtype_float=np.float32,
    ) -> dict:
        """
        Fast HDF5 loader returning data organized per file.

        Returns
        -------
        out["data"][file_index][dataset_name] = numpy array
        """

        info = self.get(int(snapnum))
        files = list(info.files)

        # select files
        if file_indices is None:
            file_indices = list(range(len(files)))
        else:
            file_indices = [int(i) for i in file_indices]

        files_sel = [files[i] for i in file_indices]

        if len(files_sel) == 0:
            raise ValueError("No files selected.")

        # determine datasets if None
        if datasets is None:
            with h5py.File(files_sel[0], "r") as f:
                g = f[part_group]
                datasets = [k for k, v in g.items() if isinstance(v, h5py.Dataset)]
            datasets = sorted(datasets)

        data = {}

        self.log.info(
            "load_snapdir_hdf5: snap=%03d group=%s nfiles=%d ndatasets=%d",
            int(snapnum), part_group, len(files_sel), len(datasets),
        )

        for fi, fn in zip(file_indices, files_sel):

            with h5py.File(fn, "r") as f:
                if part_group not in f:
                    continue

                g = f[part_group]
                data[fi] = {}

                for name in datasets:
                    dset = g.get(name, None)
                    if dset is None:
                        data[fi][name] = None
                        continue

                    arr = dset[()]

                    if dtype_float is not None and np.issubdtype(arr.dtype, np.floating):
                        arr = arr.astype(dtype_float, copy=False)

                    data[fi][name] = arr

        return {
            "snapnum": int(snapnum),
            "files": [str(files[i]) for i in file_indices],
            "group": part_group,
            "data": data,
        }

    def load_snapdir_hdf5_masked(
        self,
        snapnum: int,
        part_group: str = "PartType0",
        datasets: list[str] | None = None,
        file_indices: list[int] | None = None,
        selected_indices_by_file: dict[int, np.ndarray] | None = None,
        dtype_float=np.float32,
        sort_indices: bool = True,
    ) -> dict:
        """
        Load datasets per file and optionally mask rows.

        Returns
        -------
        out["data"][file_index][dataset_name] = numpy array
        """

        info = self.get(int(snapnum))
        files_all = list(info.files)

        if file_indices is None:
            file_indices = list(range(len(files_all)))
        else:
            file_indices = [int(i) for i in file_indices]

        if selected_indices_by_file is None:
            return self.load_snapdir_hdf5(
                snapnum,
                part_group,
                datasets,
                file_indices,
                dtype_float,
            )

        # validate keys
        for fi in file_indices:
            if fi not in selected_indices_by_file:
                raise ValueError(f"Missing indices for file {fi}")

        # determine datasets if None
        if datasets is None:
            probe = self.load_snapdir_hdf5(
                snapnum,
                part_group,
                datasets=None,
                file_indices=[file_indices[0]],
                dtype_float=dtype_float,
            )
            datasets = list(probe["data"][file_indices[0]].keys())

        data_out = {}

        self.log.info(
            "load_snapdir_hdf5_masked: snap=%03d nfiles=%d ndatasets=%d",
            int(snapnum), len(file_indices), len(datasets),
        )

        for fi in file_indices:

            idx = np.asarray(selected_indices_by_file[fi], dtype=np.int64)

            if sort_indices and idx.size > 1 and not np.all(idx[1:] >= idx[:-1]):
                idx = np.sort(idx)

            one = self.load_snapdir_hdf5(
                snapnum,
                part_group,
                datasets,
                file_indices=[fi],
                dtype_float=dtype_float,
            )

            data_out[fi] = {}

            for name in datasets:
                arr = one["data"][fi].get(name, None)
                if arr is None:
                    data_out[fi][name] = None
                    continue

                if idx.size == 0:
                    data_out[fi][name] = arr[:0]
                else:
                    data_out[fi][name] = arr[idx]

        return {
            "snapnum": int(snapnum),
            "files": [str(files_all[i]) for i in file_indices],
            "group": part_group,
            "data": data_out,
        }
    
    def get_index_cKD_neighbours(
        self,
        sample_points: np.ndarray | None,
        snapnum: int,
        Rmax_ckpc_over_h: float,
        cache_tree: bool = True,
        force_rebuild: bool = False,
    ):
        """
        Return gas-cell candidate neighbor indices per file using a fixed radius Rmax.

        If sample_points is None:
            return ALL gas cells.

        Otherwise:
            candidates are all cells with
                distance(point, cell_center) <= Rmax_ckpc_over_h
            (periodic minimal-image handled by cKDTree(boxsize=L)).

        Caching:
            If cache_tree=True, cache the KDTree + mapping arrays on self per snapnum.
        """

        info = self.get(int(snapnum))
        files = list(info.files)

        # --------------------------------------------------
        # 0) ALL cells shortcut
        # --------------------------------------------------
        if sample_points is None:
            neighbors_by_file = {}
            for fid, fn in enumerate(files):
                with h5py.File(fn, "r") as f:
                    Ni = f["PartType0"]["Coordinates"].shape[0]
                neighbors_by_file[fid] = np.arange(Ni, dtype=np.int64)
            return neighbors_by_file

        # --------------------------------------------------
        # 1) Prepare query points
        # --------------------------------------------------
        sample_points = np.asarray(sample_points, dtype=np.float32)
        if sample_points.ndim != 2 or sample_points.shape[1] != 3:
            raise ValueError("sample_points must have shape (B,3)")

        L = float(info.cosmo["BoxSize"])  # ckpc/h
        sample_points = np.mod(sample_points, L)
        sample_points[sample_points == L] = 0.0

        Rmax_ckpc_over_h = float(Rmax_ckpc_over_h)
        if Rmax_ckpc_over_h <= 0:
            raise ValueError("Rmax_ckpc_over_h must be > 0")

        self.log.info(
            "Candidate neighbor search: B=%d  Rmax_ckpc_over_h=%.3f ckpc/h  cache_tree=%s",
            sample_points.shape[0], Rmax_ckpc_over_h, cache_tree
        )

        # --------------------------------------------------
        # 2) KDTree cache (per snapshot)
        # --------------------------------------------------
        cache_key = f"_ckd_candidates_cache_snap{int(snapnum):03d}"

        # A small signature to ensure the cache matches current snapshot files (cheap but robust enough: list of file paths + BoxSize)
        signature = (tuple(str(p) for p in files), float(L))

        use_cache = (
            cache_tree
            and (not force_rebuild)
            and hasattr(self, cache_key)
            and getattr(self, cache_key).get("signature", None) == signature
        )

        if use_cache:
            cache = getattr(self, cache_key)
            tree = cache["tree"]
            file_id = cache["file_id"]
            local_index = cache["local_index"]
            # coords not strictly needed after tree build, but may be nice to keep coords = cache.get("coords", None)
            self.log.info("Using cached KDTree (%s)", cache_key)

        else:
            self.log.info("Building KDTree cache for snap %03d (loading Coordinates)", int(snapnum))

            # FAST load Coordinates, per file
            loaded = self.load_snapdir_hdf5(
                snapnum=int(snapnum), part_group="PartType0", datasets=["Coordinates"], file_indices=None, dtype_float=np.float32,
            )["data"]  # loaded[fid]["Coordinates"]

            coords_list = []
            file_id_list = []
            local_idx_list = []

            for fid in sorted(loaded.keys()):
                c = loaded[fid]["Coordinates"]  # (Ni,3) float32

                # periodic wrap once
                c = np.mod(c, L)
                c[c == L] = 0.0

                Ni = c.shape[0]
                coords_list.append(c)
                file_id_list.append(np.full(Ni, fid, dtype=np.int32))
                local_idx_list.append(np.arange(Ni, dtype=np.int64))

            coords = np.concatenate(coords_list, axis=0)
            file_id = np.concatenate(file_id_list, axis=0)
            local_index = np.concatenate(local_idx_list, axis=0)

            self.log.info("Building periodic KDTree (N=%d)", coords.shape[0])
            tree = cKDTree(coords, boxsize=L)

            if cache_tree:
                setattr(
                    self,
                    cache_key,
                    {
                        "signature": signature,
                        "tree": tree,
                        "file_id": file_id,
                        "local_index": local_index,
                        # optional: keep coords if you want (costs memory)
                        # "coords": coords,
                    },
                )
                self.log.info("Cached KDTree under %s", cache_key)

        # --------------------------------------------------
        # 3) Query candidates (fixed Rmax)
        # --------------------------------------------------
        self.log.info("Querying candidates within Rmax")
        cand_lists = tree.query_ball_point(sample_points, r=Rmax_ckpc_over_h)  # list-of-lists

        # --------------------------------------------------
        # 4) Union candidates per file
        # --------------------------------------------------
        # Flatten candidates to one array (avoid Python set updates per point) Guard against "all empty" case.
        nonempty = [c for c in cand_lists if len(c)]
        if not nonempty:
            neighbours_by_file = {fid: np.array([], dtype=np.int64) for fid in range(len(files))}
            self.log.info("Total unique candidates: 0")
            return neighbours_by_file

        flat = np.unique(np.concatenate([np.asarray(c, dtype=np.int64) for c in nonempty]))

        fids = file_id[flat]
        lids = local_index[flat]

        neighbours_by_file = {fid: np.array([], dtype=np.int64) for fid in range(len(files))}
        for fid in np.unique(fids):
            fid = int(fid)
            neighbours_by_file[fid] = np.unique(lids[fids == fid]).astype(np.int64, copy=False)

        self.log.info("Total unique candidates: %d", sum(len(v) for v in neighbours_by_file.values()))
        return neighbours_by_file


    def compute_cloudy_ion_number_densities(
        self,
        n_dict,
        rho_gas_phys,
        T_gas,
        redshift,
        cloudy_table_dir,
        ions_to_compute=[("H", 1)],
        key_element="all",
    ):
        """
        Compute ion number densities using Cloudy tables.

        Parameters
        ----------
        n_dict : dict
            Element nuclei number densities [cm^-3]
            Example: n_dict["H"]["all"], n_dict["Si"]["all"], ...
        rho_gas_phys : array
            Gas physical mass density [g/cm^3]
        T_gas : array
            Gas temperature [K]
        redshift : float
            Snapshot redshift
        cloudy_table_dir : str
            Path to Cloudy tables
        ions_to_compute : list of (element_symbol, ion_stage)
            Example: [("H",1), ("Si",2), ("Si",3)]

        Returns
        -------
        ion_number_density_dict : dict
            Nested dictionary:
                ion_number_density[element][ion_stage] = number_density [cm^-3]
        """

        # 1) Total atomic number density for Cloudy
        rho_atoms = rho_gas_phys / phys.mp  # [atoms/cm^3]

        # 2) Load / cache Cloudy table
        if not hasattr(self, "_cloudy_cache"):
            self._cloudy_cache = {}

        z = float(redshift)
        cache_key = (cloudy_table_dir, round(z, 6))

        ct = self._cloudy_cache.get(cache_key, None)
        if ct is None:
            ct = convert_cloudy.CloudyTable(z, directory=cloudy_table_dir)
            self._cloudy_cache[cache_key] = ct

        # 3) Clip inputs to table bounds
        rho_atoms_safe = np.clip(rho_atoms, 10.0 ** ct.dens[0], 10.0 ** ct.dens[-1])
        T_safe = np.clip(T_gas, 10.0 ** ct.temp[0], 10.0 ** ct.temp[-1])

        # 4) Compute ion number densities
        for elem_sym, ion_stage in ions_to_compute:
            ion_frac = ct.ion(elem_sym, ion_stage, rho_atoms_safe, T_safe) # Compute ion fraction
            n_dict[elem_sym][ion_stage] = n_dict[elem_sym][key_element] * ion_frac  # [cm^-3]

        return n_dict
    def apply_rahmati_HI_correction(
        self,
        redshift,
        nH_phys: np.ndarray,
        temperature_gas: np.ndarray,
        sfr: np.ndarray | None
    ) -> np.ndarray:
        """
        Apply Rahmati+2013 self-shielding correction to HI only (a-posteriori).

        Parameters
        ----------
        redshift : float
        nH_phys : array
            Total hydrogen number density (physical, cm^-3).  (You are passing H["all"], good.)
        temperature_gas : array
            Gas cell temperature (K), one per cell.
        sfr : array or None
            StarFormationRate field. If None, treated as zeros.

        Returns
        -------
        nHI_rah : array
            HI number density with Rahmati correction (cm^-3), physical.
        """
        zz = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.float64)
        gamma_UVB_vals = np.array([3.99e-14, 3.03e-13, 6.00e-13, 5.53e-13, 4.31e-13, 3.52e-13, 2.678e-13], dtype=np.float64)
        gray_opac_vals = np.array([2.59e-18, 2.37e-18, 2.27e-18, 2.15e-18, 2.02e-18, 1.94e-18, 1.82e-18], dtype=np.float64)

        gamma_UVB = np.interp(redshift, zz, gamma_UVB_vals)   # [s^-1]
        gray_opac = np.interp(redshift, zz, gray_opac_vals)   # [cm^2]

        fbar = float(self.cosmo["Omega_b"]) / float(self.cosmo["Omega_m"])
        T4 = np.maximum(temperature_gas / 1.0e4, 1.0e-8)
        G12 = gamma_UVB / 1.0e-12

        nSSh = (
            6.73e-3
            * (gray_opac / 2.49e-18) ** (-2.0 / 3.0)
            * (T4 ** 0.17)
            * (G12 ** (2.0 / 3.0))
            * (fbar / 0.17) ** (-1.0 / 3.0)
        )

        y = np.maximum(nH_phys / np.maximum(nSSh, 1e-60), 0.0)

        photo_rate = gamma_UVB * (
            0.98 * (1.0 + y**1.64) ** (-2.28)
            + 0.02 * (1.0 + y) ** (-0.84)
        )  # [s^-1]

        lamb = 315614.0 / np.maximum(temperature_gas, 1.0)
        alpha_A = 1.269e-13 * lamb**1.503 / (1.0 + (lamb / 0.522) ** 0.47) ** 1.923  # [cm^3/s]
        LambdaT = (
            1.17e-10
            * np.sqrt(np.maximum(temperature_gas, 0.0))
            * np.exp(-157809.0 / np.maximum(temperature_gas, 1.0))
            / (1.0 + np.sqrt(np.maximum(temperature_gas, 0.0) / 1.0e5))
        )  # [cm^3/s]

        A = alpha_A + LambdaT
        B = 2.0 * alpha_A + (photo_rate / np.maximum(nH_phys, 1e-60)) + LambdaT

        disc = np.maximum(B * B - 4.0 * A * alpha_A, 0.0)
        xHI_rah = (B - np.sqrt(disc)) / (2.0 * A)
        xHI_rah = np.clip(xHI_rah, 0.0, 1.0)

        # Star-forming gas handling
        if sfr is None:
            sfr = np.zeros_like(nH_phys)
        else:
            sfr = np.asarray(sfr)

        sfr_mask = (sfr > 0)
        if np.any(sfr_mask):
            T_SF = 1.0e4
            T4_SF = 1.0

            nSSh_SF = (
                6.73e-3
                * (gray_opac / 2.49e-18) ** (-2.0 / 3.0)
                * (T4_SF ** 0.17)
                * (G12 ** (2.0 / 3.0))
                * (fbar / 0.17) ** (-1.0 / 3.0)
            )

            y_sf = np.maximum(nH_phys[sfr_mask] / np.maximum(nSSh_SF, 1e-60), 0.0)

            photo_rate_sf = gamma_UVB * (
                0.98 * (1.0 + y_sf**1.64) ** (-2.28)
                + 0.02 * (1.0 + y_sf) ** (-0.84)
            )

            lamb_sf = 315614.0 / T_SF
            alpha_A_sf = 1.269e-13 * lamb_sf**1.503 / (1.0 + (lamb_sf / 0.522) ** 0.47) ** 1.923
            LambdaT_sf = 1.17e-10 * np.sqrt(T_SF) * np.exp(-157809.0 / T_SF) / (1.0 + np.sqrt(T_SF / 1.0e5))

            A_sf = alpha_A_sf + LambdaT_sf
            B_sf = 2.0 * alpha_A_sf + (photo_rate_sf / np.maximum(nH_phys[sfr_mask], 1e-60)) + LambdaT_sf
            disc_sf = np.maximum(B_sf * B_sf - 4.0 * A_sf * alpha_A_sf, 0.0)

            xHI_sf = (B_sf - np.sqrt(disc_sf)) / (2.0 * A_sf)
            xHI_sf = np.clip(xHI_sf, 0.0, 1.0)

            xHI_rah[sfr_mask] = xHI_sf

        return nH_phys * xHI_rah

    def load_fields_from_simulation(
            self,
            snapnum=None,
            sample_points_Mpc=None,
            ions_to_compute=[("H", 1)],
            HI_method="Rahmati",  # "Rahmati", "Cloudy", or "NeutralHydrogenAbundance"
            Rmax_ckpc_over_h=None,
            TNG_element_names_ordered=["H", "He", "C", "N", "O", "Ne", "Mg", "Si", "Fe", "Other"],
            XH_primordial=0.76,
            gamma=5/3,
            cloudy_table_subdir_name="ion_out_photo_atten",
        ):
        """
        Load (optionally sample) gas-cell fields from TNG/AREPO snapshots and derive:
        - gas temperature per cell (TNG convention from InternalEnergy + ElectronAbundance)
        - per-element total number densities (physical, cm^-3)
        - per-ion number densities from Cloudy tables (physical, cm^-3)
        - HI via one of: Rahmati / Cloudy / NeutralHydrogenAbundance

        Output structure:
        fields_from_simulation[snapnum] = {
            "Coordinates_cMpc": (N,3) float64/32,
            "number_density": dict (nested: elem -> ion/state -> (N,)),
            "temperature": {"gas": (N,)},
            "velocity": {"gas": (N,3)},
        }

        Notes:
        - Coordinates are converted to comoving Mpc (cMpc).
        - Temperature is a gas-cell property (one T per cell). Ion-weighted T(x) is computed later at sampling time.
        - Memory logging reports numpy array sizes only (not Python dict overhead).
        """

        # ----------------------------
        # Resolve snapshots
        # ----------------------------
        if snapnum is None:
            snapnums = list(self.snapnums)
        elif isinstance(snapnum, (list, tuple, np.ndarray)):
            snapnums = [int(x) for x in snapnum]
        else:
            snapnums = [int(snapnum)]

        little_h = np.float32(self.cosmo["h"])

        # ----------------------------
        # Determine required elements from ions_to_compute
        # ----------------------------
        elements_needed = {elem for elem, _ in ions_to_compute}
        valid_elements = set(TNG_element_names_ordered)
        invalid = elements_needed - valid_elements
        if invalid:
            raise ValueError(f"Requested elements not available in TNG outputs: {invalid}")

        # Atomic masses (amu -> g)
        elems_for_mass = [e for e in elements_needed if e != "Other"]
        mass_elements_amu_dict = get_mass_amu(elems_for_mass)
        mass_elements_g_dict = {e: mass_elements_amu_dict[e] / phys.g_to_amu for e in mass_elements_amu_dict}

        # Cloudy tables directory
        cloudy_table_dir = get_default_cloudy_table_dir(subdir=cloudy_table_subdir_name)

        # ----------------------------
        # Gather fields
        # ----------------------------
        fields_from_simulation = {}
        for snapnum in snapnums:

            fields_from_simulation[snapnum] = {}

            info = self.get(snapnum)
            files = list(info.files)

            _scale_factor = np.float32(info.cosmo["ScaleFactor"])

            self.log.info("Snapshot %03d: z=%s  a=%s  nfiles=%d", snapnum, info.cosmo.get("Redshift"), info.cosmo.get("ScaleFactor"), len(files))

















            # ----------------------------
            # per-file indices
            # ----------------------------

            # Rmax_ckpc_over_h is either None or a single float
            if Rmax_ckpc_over_h is None:
                self.log.info("Extracting Rmax_ckpc_over_h...")
                data = self.load_snapdir_hdf5(snapnum=snapnum, datasets=["Density", "Masses"])
                _Rmax_ckpc_over_h = []
                for _, file_id in enumerate(data['data'].keys()):
                    _Rmax_ckpc_over_h.append(np.cbrt(data['data'][file_id]['Masses'] / data['data'][file_id]['Density']))
                _Rmax_ckpc_over_h = np.concatenate(_Rmax_ckpc_over_h, axis=0)
                _Rmax_ckpc_over_h = np.max(_Rmax_ckpc_over_h)
            else:
                _Rmax_ckpc_over_h = Rmax_ckpc_over_h
            self.log.info("Rmax_ckpc_over_h=%s", _Rmax_ckpc_over_h)
            
            self.log.info("Extracting per-file indices...")
            # sample_points_Mpc is either None or an array-like (B,3) in cMpc
            if sample_points_Mpc is not None:
                sample_points_ckpc_over_h = np.asarray(sample_points_Mpc) * (1000.0 * little_h)  # cMpc -> ckpc/h
            else:
                sample_points_ckpc_over_h = None

            neighbours_by_file = self.get_index_cKD_neighbours(
                sample_points=sample_points_ckpc_over_h, snapnum=snapnum, Rmax_ckpc_over_h=Rmax_ckpc_over_h,
            )

            # ----------------------------
            # coordinates
            # ----------------------------
            self.log.info("Extracting coordinates...")

            fields_from_simulation[snapnum]["coordinates_cMpc"] = self.gather_fields_from_neighbors_by_file(
                files, "Coordinates", neighbors_by_file
            ) / (1000.0 * little_h) # ckpc/h -> cMpc

            masked = self.load_snapdir_hdf5_masked(
                snapnum=snapnum, datasets=["Coordinates"], selected_indices_by_file = neighbours_by_file
            )

            # ----------------------------
            # velocities
            # ----------------------------
            self.log.info("Extracting velocities...")
            fields_from_simulation[snapnum]["velocity_kms"] = self.gather_fields_from_neighbors_by_file(
                files, "Velocities", neighbors_by_file
            ) * np.sqrt(_scale_factor) # km/s * sqrt(a) -> km/s

            # ----------------------------
            # effective cell sizes
            # ----------------------------
            self.log.info("Extracting effective cell size...")
            dens = self.gather_fields_from_neighbors_by_file(
                files, "Density", neighbors_by_file
            ) # code density (comoving) [10^10 M_\odot h^2 / ckpc^3]
            mass = self.gather_fields_from_neighbors_by_file(
                files, "Masses", neighbors_by_file
            ) # code mass (comoving) [10^10 M_\odot / h]

            # Convert (ckpc/h)^3 -> (cMpc)^3 then compute the effective cell size cMpc
            fields_from_simulation[snapnum]["radius_cMpc"] = np.cbrt(mass / dens * 1e-9 / (little_h**3))

            # ----------------------------
            # temperatures
            # ----------------------------
            self.log.info("Extracting temperatures...")
            u_cgs = self.gather_fields_from_neighbors_by_file(files, "InternalEnergy", neighbors_by_file) * (info.header["UnitVelocity_in_cm_per_s"] ** 2)  # erg/g
            mu = 4.0 / (1.0 + 3.0 * XH_primordial + 4.0 * XH_primordial * self.gather_fields_from_neighbors_by_file(files, "ElectronAbundance", neighbors_by_file))
            fields_from_simulation[snapnum]["temperature_K"] = (gamma - 1.0) * (mu * phys.mp / phys.kB_erg_per_K) * u_cgs  # K

            # ----------------------------
            # per-element number densities (physical)
            # ----------------------------
            self.log.info("Extracting number densities...")
            elemfr = self.gather_fields_from_neighbors_by_file(files, "GFM_Metals", neighbors_by_file)

            density_unit_cgs = info.header["UnitMass_in_g"] / (info.header["UnitLength_in_cm"] ** 3)  # g/cm^3
            rho_gas_phys = dens * density_unit_cgs * (_scale_factor ** -3)  # physical g/cm^3

            number_density_dict = {"gas": rho_gas_phys / phys.mp}
            for elem in elements_needed:
                j = TNG_element_names_ordered.index(elem)
                number_density_dict[elem] = {}
                number_density_dict[elem]["all"] = rho_gas_phys * elemfr[:, j] / mass_elements_g_dict[elem]

            # ----------------------------
            # per-ion number densities (Cloudy)
            # ----------------------------
            self.log.info("Computing ion number densities...")

            number_density_dict = self.compute_cloudy_ion_number_densities(
                n_dict=number_density_dict,
                rho_gas_phys=rho_gas_phys,
                T_gas=fields_from_simulation[snapnum]["temperature_K"],
                redshift=info.cosmo["Redshift"],
                cloudy_table_dir=cloudy_table_dir,
                ions_to_compute=ions_to_compute,
            )

            # ----------------------------
            # HI channel selection
            # ----------------------------
            self.log.info("Computing HI channel...")
            if "H" in number_density_dict:
                if HI_method == "NeutralHydrogenAbundance":
                    number_density_dict["H"][1] = number_density_dict["H"]["all"] * self.gather_fields_from_neighbors_by_file(files, "NeutralHydrogenAbundance", neighbors_by_file)
                elif HI_method == "Rahmati":
                    number_density_dict["H"][1] = self.apply_rahmati_HI_correction(
                        redshift=info.cosmo["Redshift"], nH_phys=number_density_dict["H"]["all"],
                        temperature_gas=fields_from_simulation[snapnum]["temperature_K"],
                        sfr=self.gather_fields_from_neighbors_by_file(files, "StarFormationRate", neighbors_by_file),
                    )
                elif HI_method == "Cloudy":
                    # assume compute_cloudy_ion_number_densities already populated H/I
                    if ("I" not in number_density_dict["H"]) and (1 not in number_density_dict["H"]):
                        self.log.warning("HI_method=Cloudy but no HI key found under number_density_dict['H'].")
                else:
                    self.log.warning("Invalid HI_method=%s. Supported: 'NeutralHydrogenAbundance', 'Rahmati', 'Cloudy'.", HI_method)

            fields_from_simulation[snapnum]["number_density_cm3"] = number_density_dict

        return fields_from_simulation

