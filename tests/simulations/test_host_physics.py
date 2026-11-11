"""The fused host physics of the streaming route (``simulations.host_physics``) gives the values of
``snapshot_cells.gas_to_cells`` for every option it covers (synthetic TNG-like cells: diffuse, star-forming, dense,
beyond the CLOUDY table)."""

from types import SimpleNamespace

import numpy as np
import pytest

from skewerjax.simulations import host_physics as hp
from skewerjax.simulations.gas_cells import GasCells
from skewerjax.simulations.snapshot_cells import GAS_SPECIES, gas_to_cells

SPECIES = ["HI", "SiIII", "CIV"]
OPTIONS = [
    dict(),
    dict(x_HI_model="stored"),
    dict(
        mass_unit="proton",
        T_constants="fake_spectra",
        x_HI_model="fake_sf",
        cloudy_density="fake_spectra",
        cloudy_interpolation="cubic",
    ),
    dict(cloudy_interpolation="cubic"),
    dict(cloudy_density="fake_spectra"),
]


def _raw(n=20000, seed=4):
    rng = np.random.default_rng(seed)
    met = rng.uniform(0, 0.02, (n, 10)).astype(np.float32)
    met[:, 0] = rng.uniform(0.70, 0.76, n)
    met[::97, 7] = -1e-9  # negative round-off: clipped to 0
    raw = {
        "Density": (10 ** rng.uniform(-6, 4, n)).astype(np.float32),  # code units: dense and diffuse, beyond the table
        "InternalEnergy": (10 ** rng.uniform(0, 7, n)).astype(np.float32),
        "ElectronAbundance": rng.uniform(0, 1.2, n).astype(np.float32),
        "NeutralHydrogenAbundance": (10 ** rng.uniform(-7, 0, n)).astype(np.float32),
        "GFM_Metals": met,
        "StarFormationRate": np.where(rng.uniform(size=n) < 0.1, rng.uniform(0, 1, n), 0.0).astype(np.float32),
        "Velocities": rng.normal(0, 200, (n, 3)).astype(np.float32),
    }
    a = 0.25
    header = SimpleNamespace(a=a, density_cgs=6.77e-31 * 0.6774**2 / a**3, unit_velocity_cms=1e5)
    return raw, header


def _gas(raw, header):
    n = raw["Density"].size
    f = dict(
        rho=raw["Density"].astype(np.float64) * header.density_cgs,
        u=raw["InternalEnergy"].astype(np.float64) * header.unit_velocity_cms**2,
        n_e=raw["ElectronAbundance"].astype(np.float64),
        x_HI_stored=raw["NeutralHydrogenAbundance"].astype(np.float64),
        metals=np.clip(raw["GFM_Metals"].astype(np.float64)[:, :9], 0.0, None),
        sfr=raw["StarFormationRate"].astype(np.float64),
        velocity=raw["Velocities"].astype(np.float64) * np.sqrt(header.a),
    )
    return GasCells(np.zeros((n, 3)), np.ones(n), f, box=10.0, a=header.a)


@pytest.mark.parametrize("opts", OPTIONS)
def test_fused_physics_equals_gas_to_cells(opts):
    raw, header = _raw()
    ref = gas_to_cells(_gas(raw, header), SPECIES, **opts)
    got = hp.species_fields(raw, header, SPECIES, **opts)
    for sp in SPECIES:
        n_ref = ref.species[sp]["number_density"].astype(np.float32)
        T_ref = ref.species[sp]["temperature"].astype(np.float32)
        n, T = got[sp]
        assert n.dtype == np.float32 and T.dtype == np.float32
        np.testing.assert_allclose(n, n_ref, rtol=2e-7, atol=0)  # float32: at most one unit in the last place
        np.testing.assert_allclose(T, T_ref, rtol=2e-7, atol=0)


def test_unsupported_options_fall_back():
    assert hp.supported(["HI"]) and not hp.supported(["HI"], x_HI_model="rahmati_all") and not hp.supported([GAS_SPECIES])
    raw, header = _raw(100)
    with pytest.raises(ValueError):
        hp.species_fields(raw, header, ["HI"], x_HI_model="rahmati_all")
    raw["Density"][3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        hp.species_fields(raw, header, ["HI"])
