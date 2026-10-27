"""Per-cell gas physics (notebook 05, layers 2-3 and 5) on synthetic cells, without a snapshot: ``Ion`` and
``ion_from_species``, the element densities (mass units), T(u, n_e), the Doppler b, the H I models (stored,
star-forming cells recomputed, Rahmati+13 limits), the CLOUDY table (exact at its nodes, clamped, redshift range),
``ion_properties`` for H I and a metal ion, ``GasCells`` (take, concatenate, pytree) and ``gas_to_cells`` (the
``core.Cells`` of the snapshot: n_com = n_ion a^3). The same physics on TNG50-4: ``test_snapshot.py``,
``tests/simulations/test_snapshot.py``."""

import jax
import numpy as np
import pytest

from skewerjax.core import phys
from skewerjax.simulations import HI, CloudyIonTable, GasCells, Ion, gas_to_cells, ion_from_species, ion_properties
from skewerjax.simulations import gas_physics as gp

A, N = 0.25, 64


def synthetic(seed=0):
    """N cells spanning the n_H-T plane of a z = 3 snapshot, a quarter of them star-forming."""
    rng = np.random.default_rng(seed)
    rho = 10 ** rng.uniform(-30, -23, N)  # [g cm^-3]
    metals = np.zeros((N, 9))
    metals[:, 0], metals[:, 1], metals[:, 7] = 0.76, 0.24, 1e-4
    f = dict(
        rho=rho,
        u=10 ** rng.uniform(11.5, 15, N),
        n_e=rng.uniform(1.0, 1.16, N),
        x_HI_stored=10 ** rng.uniform(-6, -1, N),
        metals=metals,
        sfr=np.where(np.arange(N) % 4 == 0, 1.0, 0.0),
        velocity=rng.normal(0, 100, (N, 3)),
    )
    return GasCells(rng.uniform(0, 10, (N, 3)), 10 ** rng.uniform(-6, -3, N), f, box=10.0, a=A, ids=np.arange(N))


def test_ions():
    si = ion_from_species("SiIII")
    assert (si.element, si.stage, si.species, si.metals_column, si.is_HI) == ("Si", 3, "SiIII", 7, False)
    assert ion_from_species("HI").is_HI and HI.mass_amu == 1.00794 and Ion("C", 4).species == "CIV"
    for bad in ("Si3", "XxII", "HIIIII"):
        with pytest.raises(ValueError):
            ion_from_species(bad)
    with pytest.raises(ValueError):
        Ion("Si", 0)


def test_densities_temperature_and_b():
    rho, X = np.array([1e-27, 1e-25]), np.array([0.76, 0.76])
    n = np.asarray(gp.element_number_density(rho, X, 1.00794))
    np.testing.assert_allclose(n, rho * X / (1.00794 * gp.M_U_G), rtol=1e-14)
    n_p = np.asarray(gp.element_number_density(rho, X, 1.00794, mass_unit="proton"))
    np.testing.assert_allclose(n / n_p, phys.mp / gp.M_U_G, rtol=1e-14)  # m_p / m_u = 1.00728
    with pytest.raises(ValueError):
        gp.element_number_density(rho, X, 1.0, mass_unit="kg")
    u, ne = np.array([1e13, 1e14, -1.0]), np.array([1.1, 0.0, 1.0])
    T = np.asarray(gp.temperature(u, ne))
    mu = 4.0 / (1.0 + 3 * 0.76 + 4 * 0.76 * ne[:2])
    np.testing.assert_allclose(T[:2], (2.0 / 3.0) * mu * phys.mp / phys.kB_erg_per_K * u[:2], rtol=1e-14)
    assert T[2] == 1.0  # T <= 0 -> 1 K
    assert abs(float(gp.temperature(1e13, 1.1, constants="fake_spectra")) / T[0] - 1 + 8.1e-6) < 1e-6
    b = float(gp.doppler_b_kms(1e4, 1.00794))
    assert abs(b - np.sqrt(2 * 1.380649e-23 * 1e4 / (1.00794 * 1.66053906660e-27)) / 1e3) < 1e-6 * b  # 12.85 km/s


def test_hydrogen_neutral_fraction():
    z = 1 / A - 1
    n_H = np.geomspace(1e-6, 1e2, 50)
    x = np.asarray(gp.rahmati_neutral_fraction(n_H, np.full(50, 1e4), z))
    assert np.all(np.diff(x) > 0) and x[0] < 1e-4 and x[-1] > 0.9  # optically thin -> self-shielded
    lam = 315614.0 / 1e4  # thin limit: x = alpha_A n_H / Gamma
    alpha = 1.269e-13 * lam**1.503 / (1.0 + (lam / 0.522) ** 0.47) ** 1.923
    gamma = float(np.interp(z, gp._RAHMATI_Z, gp._RAHMATI_GAMMA_FG09))
    assert abs(x[0] / (alpha * n_H[0] / gamma) - 1) < 0.02
    rates = gp.treecool_rates(z, gp.default_treecool_path())
    assert 1e-13 < rates["Gamma_HI"] < 1e-11
    with pytest.raises(ValueError):
        gp.treecool_rates(20.0, gp.default_treecool_path())
    c = synthetic()
    sf = c.fields["sfr"] > 0
    P = ion_properties(c, HI)
    x_def = np.asarray(P["ion_fraction"])
    np.testing.assert_array_equal(x_def[~sf], c.fields["x_HI_stored"][~sf])  # stored outside star-forming cells
    x_cold = np.asarray(gp.rahmati_neutral_fraction(P["n_H"], 1e4 * np.ones(N), z, gamma_uvb=rates["Gamma_HI"]))
    np.testing.assert_allclose(x_def[sf], x_cold[sf], rtol=1e-12)  # star-forming: the cold phase, at 1e4 K
    np.testing.assert_array_equal(np.asarray(ion_properties(c, HI, x_HI_model="stored")["ion_fraction"]), c.fields["x_HI_stored"])
    assert abs(gp.fake_spectra_sf_threshold() - 0.1199) < 5e-4  # fake_spectra's threshold for TNG50 [cm^-3]
    for model in ("fake_sf", "rahmati_all"):
        assert np.all(np.isfinite(np.asarray(ion_properties(c, HI, x_HI_model=model)["ion_fraction"])))
    with pytest.raises(ValueError):
        ion_properties(c, HI, x_HI_model="equilibrium")
    np.testing.assert_allclose(np.asarray(P["n_ion"]), np.asarray(P["n_elem"]) * x_def, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(P["n_H"]), c.fields["rho"] * 0.76 / (1.00794 * gp.M_U_G), rtol=1e-14)


def test_cloudy_table_and_metal_ions():
    tab = CloudyIonTable(3.0)
    si = Ion("Si", 3)
    grid = tab.log_fraction_grid(si)
    i, j = np.array([5, 20, 40]), np.array([10, 30, 60])
    f = np.asarray(tab.fraction(si, 10 ** tab.log_dens[i], 10 ** tab.log_temp[j]))
    np.testing.assert_allclose(np.log10(f), grid[i, j], rtol=0, atol=1e-10)  # exact at the nodes
    edge = np.asarray(tab.fraction(si, np.array([1e-12, 1e8]), np.array([1e4, 1e4])))
    np.testing.assert_allclose(edge, np.asarray(tab.fraction(si, 10 ** tab.log_dens[[0, -1]], np.array([1e4, 1e4]))), rtol=1e-10)
    np.testing.assert_allclose(
        np.log10(np.asarray(tab.fraction(si, 10 ** tab.log_dens[i], 10 ** tab.log_temp[j], interpolation="cubic"))),
        grid[i, j],
        atol=1e-6,
    )
    with pytest.raises(ValueError):
        tab.fraction(si, 1e-3, 1e4, interpolation="nearest")
    with pytest.raises(ValueError):
        CloudyIonTable(12.0)
    c = synthetic(1)
    P = ion_properties(c, si, cloudy=tab)
    n_H = np.asarray(P["n_H"])
    np.testing.assert_allclose(np.asarray(P["ion_fraction"]), np.asarray(tab.fraction(si, n_H, P["T"])), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(P["n_elem"]), c.fields["rho"] * 1e-4 / (28.0855 * gp.M_U_G), rtol=1e-14)
    Pf = ion_properties(c, si, cloudy=tab, cloudy_density="fake_spectra")  # the matched option: 0.774132 rho / m_p
    hden = gp.CLOUDY_HDEN_FACTOR * np.clip(
        c.fields["rho"] / gp.CONSTANTS["fake_spectra"][0], 10 ** tab.log_dens[0], 10 ** tab.log_dens[-1]
    )
    np.testing.assert_allclose(np.asarray(Pf["ion_fraction"]), np.asarray(tab.fraction(si, hden, P["T"])), rtol=1e-12)


def test_gas_cells_and_gas_to_cells():
    c = synthetic(2)
    leaves, tree = jax.tree_util.tree_flatten(c)
    assert jax.tree_util.tree_unflatten(tree, leaves).a == A
    parts = GasCells.concatenate([c.take(np.arange(10)), c.take(np.arange(10, N))])
    assert np.array_equal(parts.ids, c.ids) and np.array_equal(parts.fields["metals"], c.fields["metals"])
    np.testing.assert_allclose(c.size_cMpc, np.cbrt(c.volumes))
    with pytest.raises(ValueError):
        GasCells.concatenate([c, GasCells(c.positions, c.volumes, {"rho": c.fields["rho"]}, 10.0, A)])  # other fields
    with pytest.raises(ValueError):
        GasCells(c.positions, c.volumes[:3], {}, 10.0, A)
    cells = gas_to_cells(c, ["HI", Ion("Si", 3)])
    assert cells.species_keys == ("HI", "SiIII") and cells.box == 10.0 and np.array_equal(cells.ids, c.ids)
    np.testing.assert_array_equal(cells.positions, c.positions)
    np.testing.assert_array_equal(cells.volumes, c.volumes)
    for sp, ion in (("HI", HI), ("SiIII", Ion("Si", 3))):
        P = ion_properties(c, ion)
        np.testing.assert_allclose(cells.species[sp]["number_density"], np.asarray(P["n_ion"]) * A**3, rtol=1e-14)  # comoving
        np.testing.assert_allclose(cells.species[sp]["temperature"], np.asarray(P["T"]), rtol=1e-14)
        np.testing.assert_array_equal(cells.species[sp]["velocity"], c.fields["velocity"])
