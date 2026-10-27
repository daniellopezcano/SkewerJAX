"""Constants and atomic data (notebook 01): ``phys`` against CODATA (``scipy.constants``; to 1e-8, CODATA 2018 vs
2022), the derived H0 and the tau prefactor C = q_e^2 / (4 eps_0 m_e c); the line table (``LINES``, ``get_line``), ``LineBundle`` (per-line arrays,
species bookkeeping, delta_v_j = c ln(lambda_ref / lambda_j), pytree through jit) and ``get_mass_amu``."""

import jax
import numpy as np
import pytest
from scipy import constants as sc

from skewerjax.core import LINES, LineBundle, get_line, get_mass_amu, phys


def test_constants_against_codata():
    close = lambda a, b, rtol: np.isclose(a, b, rtol=rtol, atol=0)
    assert phys.c == sc.c and phys.c_kms == sc.c / 1e3
    assert close(phys.qe, sc.e, 1e-8) and close(phys.me, sc.m_e, 1e-7) and close(phys.epsilon_0, sc.epsilon_0, 1e-8)
    assert close(phys.mp, 1e3 * sc.m_p, 1e-8) and close(phys.kB_erg_per_K, 1e7 * sc.k, 1e-12)
    assert close(phys.kB, sc.k / sc.atomic_mass, 1e-8)  # k_B / m_u [m^2 s^-2 K^-1]
    assert close(1.0 / phys.g_to_amu, 1e3 * sc.atomic_mass, 1e-8)
    assert close(phys.Mpc_to_m, 1e6 * sc.parsec, 1e-6) and close(phys.Gpc_to_m, 1e3 * phys.Mpc_to_m, 1e-15)
    assert phys.Gyr_to_s == 365 * 86400 * 1e9
    assert close(phys.H0, phys.H0_km_s_mpc * 1e3 / phys.Mpc_to_m * phys.Gyr_to_s, 1e-12)  # [1/Gyr]
    assert close(phys.H0_over_c, phys.H0_km_s_mpc / phys.c_kms, 1e-12)  # [1/Mpc]
    assert close(phys.cross_section_prefactor, 0.026540e-9, 1e-4)  # pi e^2 / (m_e c) = 0.02654 cm^2 Hz, per ns


def test_line_table_and_bundle():
    lya = get_line("HI-LyA")
    assert (lya.species, lya.lambda0_A, lya.f_osc) == ("HI", 1215.6701, 0.4164) and lya.A21_GHz == lya.A21_Hz * 1e-9
    assert all(l.id == k for k, l in LINES.items())
    with pytest.raises(KeyError, match="Unknown line"):
        get_line("HI-LyQ")
    with pytest.raises(ValueError):
        LineBundle(["HI-LyA", "nope"])
    ids = ["HI-LyA", "SiIII-1206", "CIV-1548", "CIV-1551"]
    lb = LineBundle(ids, ref_line_id="SiIII-1206")
    assert lb.n_lines == 4 and lb.species_keys == ["HI", "SiIII", "CIV"] and lb.n_species == 3
    assert lb.indices_for_species("CIV") == [2, 3] and [l.id for l in lb.lines_for_species("CIV")] == ids[2:]
    assert lb.ref_index() == 1 and LineBundle(["HI-LyA"], ref_line_id="SiIII-1206").ref_index() is None
    lam = np.array([LINES[i].lambda0_A for i in ids])
    np.testing.assert_array_equal(np.asarray(lb.lambda0_A), lam)
    np.testing.assert_array_equal(np.asarray(lb.f_osc), [LINES[i].f_osc for i in ids])
    np.testing.assert_array_equal(np.asarray(lb.mass_amu), [LINES[i].mass_amu for i in ids])
    dv = phys.c_kms * np.log(LINES["SiIII-1206"].lambda0_A / lam)
    np.testing.assert_allclose(np.asarray(lb.delta_v), dv, rtol=1e-14)
    assert lb.delta_v[1] == 0.0 and np.isclose(lb.delta_v_for_line("HI-LyA"), dv[0], rtol=1e-14)
    with pytest.raises(KeyError):
        lb.delta_v_for_line("HI-LyB")
    out = jax.jit(lambda b: b.delta_v)(lb)  # a pytree through jit
    np.testing.assert_allclose(np.asarray(out), dv, rtol=1e-14)


def test_atomic_masses():
    m = get_mass_amu(["h", "Si", "C"])  # case-insensitive
    assert m == {"H": 1.00794, "Si": 28.0855, "C": 12.011}
    with pytest.raises(KeyError):
        get_mass_amu(["Xx"])
