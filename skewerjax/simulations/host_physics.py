"""The gas physics of ``gas_physics`` fused for the host: the same values, one loop per block of cells.

``snapshot_cells.gas_to_cells`` evaluates the gas physics as NumPy expressions: every operation writes a temporary
array, so many threads converting blocks of cells at once are limited by memory traffic and the interpreter (2.6x on 8
threads, no gain beyond; ``docs/performance.md``, D1). The streaming route (``simulations.stream``) calls
``species_fields`` instead: the elementwise chains (units, n_X, T, the H I fraction with Rahmati+13, the coordinates in
the CLOUDY table, n a^3) are numba loops without the GIL, and the table interpolation is SciPy's ``map_coordinates`` (it
releases the GIL), so the threads of the pass scale. Every formula is the one of ``gas_physics``, in float64; the results
agree with ``gas_to_cells`` to rounding (``tests/simulations/test_host_physics.py``).

Supported: every option of ``gas_to_cells`` for H I and the metal ions, x_HI_model "stored", "stored_sf" and "fake_sf";
anything else (``GAS_SPECIES``, "rahmati_all") returns None and the caller takes ``gas_to_cells``.
"""

from __future__ import annotations

import numpy as np

from skewerjax.simulations import gas_physics as gp

_KERNELS = {}


def _kernels():
    """The numba kernels, compiled on first use (no disk cache)."""
    if _KERNELS:
        return _KERNELS
    import numba

    @numba.njit(nogil=True, inline="always")
    def rahmati(n_H, T, gamma, gray, f_bar):  # gas_physics.rahmati_neutral_fraction, one cell
        T = max(T, 1.0)
        n_ssh = (
            6.73e-3
            * (gray / 2.49e-18) ** (-2.0 / 3.0)
            * (T / 1e4) ** 0.17
            * (gamma / 1e-12) ** (2.0 / 3.0)
            * (f_bar / 0.17) ** (-1.0 / 3.0)
        )
        y = n_H / n_ssh
        photo = gamma * (0.98 * (1.0 + y**1.64) ** -2.28 + 0.02 * (1.0 + y) ** -0.84)
        lam = 315614.0 / T
        alpha_A = 1.269e-13 * lam**1.503 / (1.0 + (lam / 0.522) ** 0.47) ** 1.923
        coll = 1.17e-10 * np.sqrt(T) * np.exp(-157809.0 / T) / (1.0 + np.sqrt(T / 1e5))
        A = alpha_A + coll
        B = 2.0 * alpha_A + photo / max(n_H, 1e-300) + coll
        x = (B - np.sqrt(max(B * B - 4.0 * A * alpha_A, 0.0))) / (2.0 * A)
        return min(max(x, 0.0), 1.0)

    @numba.njit(nogil=True)
    def base(dens, ie, ea, zH, rho_cgs, u_cgs, mH_g, T_fac):
        """rho [g cm^-3], n_H [cm^-3], T [K] (``to_gas_cells``, ``element_number_density``, ``temperature``)."""
        n = dens.size
        rho, nH, T = np.empty(n), np.empty(n), np.empty(n)
        for i in range(n):
            r = np.float64(dens[i]) * rho_cgs
            rho[i] = r
            nH[i] = r / mH_g * max(np.float64(zH[i]), 0.0)
            mu = 4.0 / (1.0 + 3.0 * 0.76 + 4.0 * 0.76 * np.float64(ea[i]))
            t = T_fac * mu * (np.float64(ie[i]) * u_cgs)
            T[i] = t if t > 0 else 1.0
        return rho, nH, T

    @numba.njit(nogil=True)
    def hi_density(rho, nH, xst, sfr, mode, mp_fs, thr, gamma, gray, f_bar, a3):
        """n_HI a^3 (float32): the stored fraction, recomputed (Rahmati+13 at 1e4 K) in star-forming cells (mode 1) or
        above fake_spectra's density threshold (mode 2, n = rho / m_p) (``neutral_fraction``)."""
        n = rho.size
        out = np.empty(n, np.float32)
        for i in range(n):
            x = np.float64(xst[i])
            if mode == 1 and sfr[i] > 0:
                x = rahmati(nH[i], 1e4, gamma, gray, f_bar)
            elif mode == 2:
                r = rho[i] / mp_fs
                if r > thr / 0.76:
                    x = rahmati(r, 1e4, gamma, gray, f_bar)
            out[i] = nH[i] * x * a3
        return out

    @numba.njit(nogil=True)
    def coords(rho, nH, T, mode, clamp, mp_fs, factor, d_lo, d_hi, ld0, dld, t_lo, t_hi, lt0, dlt):
        """The coordinates of every cell in the CLOUDY table (``CloudyIonTable._coords``; mode 0: hden = n_H, 1:
        fake_spectra's 0.774132 x clip(rho / m_p))."""
        n = rho.size
        xy = np.empty((2, n))
        for i in range(n):
            if mode == 0:
                h = nH[i]
            else:
                h = factor * min(max(rho[i] / mp_fs, d_lo), d_hi)
            if clamp:
                h = min(max(h, d_lo), d_hi)
            xy[0, i] = (np.log10(h) - ld0) / dld
            xy[1, i] = (np.log10(min(max(T[i], t_lo), t_hi)) - lt0) / dlt
        return xy

    @numba.njit(nogil=True)
    def metal_density(rho, z, m_g, logf, a3):
        """n_ion a^3 (float32) = rho Z / (A m) 10^logf a^3."""
        n = rho.size
        out = np.empty(n, np.float32)
        for i in range(n):
            out[i] = rho[i] / m_g * max(np.float64(z[i]), 0.0) * 10.0 ** logf[i] * a3
        return out

    _KERNELS.update(base=base, hi=hi_density, coords=coords, metal=metal_density)
    return _KERNELS


def supported(species, x_HI_model: str = "stored_sf", **_) -> bool:
    """Whether ``species_fields`` covers these species and options."""
    from skewerjax.simulations.snapshot_cells import GAS_SPECIES

    return x_HI_model in ("stored", "stored_sf", "fake_sf") and GAS_SPECIES not in list(species)


def species_fields(
    raw: dict,
    header,
    species,
    *,
    mass_unit: str = "amu",
    T_constants: str = "codata",
    x_HI_model: str = "stored_sf",
    cloudy_density: str = "hydrogen",
    cloudy_interpolation: str = "linear",
) -> dict:
    """{species: (n a^3 [comoving cm^-3], T [K]) float32} of the raw TNG datasets ``raw`` (``Density``,
    ``InternalEnergy``, ``ElectronAbundance``, ``NeutralHydrogenAbundance``, ``GFM_Metals``, ``StarFormationRate``) of a
    snapshot with ``header``: the values of ``snapshot_cells.gas_to_cells(reader.to_gas_cells(raw), species, **options)``
    (module docstring). Raises ``ValueError`` on non-finite or negative values, as ``core.Cells`` does."""
    from scipy.ndimage import map_coordinates

    if not supported(species, x_HI_model):
        raise ValueError(f"species_fields does not cover {list(species)} with x_HI_model={x_HI_model!r}: use gas_to_cells.")
    if cloudy_interpolation not in ("linear", "cubic"):
        raise ValueError(f"interpolation must be 'linear' or 'cubic', got {cloudy_interpolation!r}.")
    if cloudy_density not in ("hydrogen", "fake_spectra"):
        raise ValueError(f"cloudy_density must be 'hydrogen' or 'fake_spectra', got {cloudy_density!r}.")
    k = _kernels()
    a = float(header.a)
    a3 = a**3
    z = 1.0 / a - 1.0
    unit_g = gp._mass_unit_g(mass_unit)
    m_p, k_B = gp.CONSTANTS[T_constants]
    T_fac = (5.0 / 3.0 - 1.0) * m_p / k_B
    met = np.asarray(raw["GFM_Metals"])
    rho, nH, T = k["base"](
        np.ascontiguousarray(raw["Density"]),
        np.ascontiguousarray(raw["InternalEnergy"]),
        np.ascontiguousarray(raw["ElectronAbundance"]),
        np.ascontiguousarray(met[:, 0]),
        float(header.density_cgs),
        float(header.unit_velocity_cms) ** 2,
        gp.get_mass_amu(["H"])["H"] * unit_g,
        T_fac,
    )
    mp_fs = gp.CONSTANTS["fake_spectra"][0]
    T32 = T.astype(np.float32)
    out, cloudy = {}, None
    for sp in species:
        ion = sp if isinstance(sp, gp.Ion) else gp.ion_from_species(sp)
        if ion.is_HI:
            mode = {"stored": 0, "stored_sf": 1, "fake_sf": 2}[x_HI_model]
            gamma = (
                gp.treecool_rates(z, gp.default_treecool_path())["Gamma_HI"]
                if mode == 1
                else float(np.interp(z, gp._RAHMATI_Z, gp._RAHMATI_GAMMA_FG09))
            )
            gray = float(np.interp(z, gp._RAHMATI_Z, gp._RAHMATI_GRAY))
            thr = gp.fake_spectra_sf_threshold() if mode == 2 else 0.0
            if ion.mass_amu != gp.get_mass_amu(["H"])["H"]:
                raise AssertionError("H I: the element mass of n_H")
            n = k["hi"](
                rho,
                nH,
                np.ascontiguousarray(raw["NeutralHydrogenAbundance"]),
                np.ascontiguousarray(raw["StarFormationRate"]),
                mode,
                mp_fs,
                thr,
                gamma,
                gray,
                0.17,
                a3,
            )
        else:
            if cloudy is None:
                cloudy = gp.CloudyIonTable(z)
            ld, lt = cloudy.log_dens, cloudy.log_temp
            xy = k["coords"](
                rho,
                nH,
                T,
                0 if cloudy_density == "hydrogen" else 1,
                cloudy_interpolation == "linear",
                mp_fs,
                gp.CLOUDY_HDEN_FACTOR,
                10 ** ld[0],
                10 ** ld[-1],
                ld[0],
                ld[1] - ld[0],
                10 ** lt[0],
                10 ** lt[-1],
                lt[0],
                lt[1] - lt[0],
            )
            logf = map_coordinates(
                cloudy.log_fraction_grid(ion), xy, order=1 if cloudy_interpolation == "linear" else 3, mode="nearest"
            )
            n = k["metal"](rho, np.ascontiguousarray(met[:, ion.metals_column]), ion.mass_amu * unit_g, logf, a3)
        bad = ~np.isfinite(n) | (n < 0)
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            raise ValueError(
                f"Cells: species {ion.species!r}, number_density: {int(bad.sum())} non-finite or negative values, the first {n[i]!r}."
            )
        out[ion.species] = (n, T32)
    return out
