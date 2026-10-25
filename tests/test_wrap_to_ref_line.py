"""Sign of the reference-line shift (``core.wrap_to_ref_line``, ``core.velocities_in_line_frames``,
``compute_optical_depth(line_frame="reference")``): a feature at v0 in a line's own frame lands at v0 - delta_v on the
reference frame, delta_v = c ln(lambda_ref / lambda_j). The shift is a change of axis: it wraps only with
``velocity_period``.

The same checks are in ``scripts/validation/check_line_frames.py``."""
import numpy as np
import jax.numpy as jnp

from skewerjax.core import (LineBundle, SkewerBundle, cells_from_values, compute_optical_depth, phys, velocities_in_line_frames,
                            wrap_to_ref_line)


def test_delta_function():
    """A feature at v0 in Si III's own frame must land at v0 - delta_v in the Lya frame."""
    LB = LineBundle(["HI-LyA", "SiIII-1206"], ref_line_id="HI-LyA")
    v = (np.arange(4000) + 0.5) * 1.0
    tau = np.zeros((2, 1, v.size))
    tau[:, 0, 3000] = 1.0
    w = np.asarray(wrap_to_ref_line(tau, v, LB))
    expected = v[3000] - float(LB.delta_v[1])
    got = v[np.nanargmax(w[1, 0])]
    assert abs(got - expected) <= 1.0, f"Si III at {got} km/s, expected {expected:.1f} (v0 - delta_v)"
    assert np.all(np.isnan(w[1, 0][v + float(LB.delta_v[1]) > v[-1]])), "no wrap: unknown off the grid"
    wp = np.asarray(wrap_to_ref_line(np.roll(tau, -2900, axis=-1), v, LB, velocity_period=4000.0))
    assert abs(v[np.argmax(wp[1, 0])] - np.mod(v[100] - float(LB.delta_v[1]), 4000.0)) <= 1.0, "periodic: modulo P"


def test_physical_absorber():
    """One static absorber seen in Lya and in an H I line at 1206.5 A, each in its own frame, then on the Lya frame:
    the 1206.5 line appears c ln(1215.67 / 1206.5) bluer, as the equality of observed wavelengths requires."""
    LB = LineBundle(["HI-LyA", "HI-LyA-1206"], ref_line_id="HI-LyA")
    a, n_pt = 0.25, 401
    pos = np.zeros((1, n_pt, 3))
    pos[0, :, 0] = np.linspace(9.9, 10.1, n_pt)
    los = np.zeros((1, n_pt, 3))
    los[..., 0] = -1.0
    sb = SkewerBundle(spatial_points=jnp.asarray(pos), a0=a, ICs_defined_at_index=0, los_directions=jnp.asarray(los),
                      manual_scale_factors=jnp.full((1, 1, n_pt), a))
    dens = np.where(np.abs(pos[..., 0] - 10.0) < 0.01, 1e-12, 0.0)
    fields = cells_from_values(pos[0], None, {"HI": dens[0]}, temperature=1e4, volumes=np.full(n_pt, (0.2 / (n_pt - 1)) ** 3))
    v = jnp.asarray((np.arange(-3000, 1000) + 0.5) * 1.0)          # wide enough for the shift: no wrap needed
    t = np.asarray(compute_optical_depth(sb, v, LB, fields, a_dagger=a))
    v_ly, v_own = float(v[np.argmax(t[0, 0])]), float(v[np.argmax(t[1, 0])])
    v_ref = float(v[np.nanargmax(np.asarray(wrap_to_ref_line(t, v, LB))[1, 0])])
    expected = v_ly + phys.c_kms * np.log(1206.5 / 1215.6701)
    assert abs(v_own - v_ly) <= 1.0, "both lines of the same gas peak at the same line-centred velocity"
    assert abs(v_ref - expected) <= 1.0, f"1206 line at {v_ref} km/s, expected {expected:.1f}"
    t_ref = np.asarray(compute_optical_depth(sb, v, LB, fields, a_dagger=a, line_frame="reference"))
    assert abs(float(v[np.argmax(t_ref[1, 0])]) - expected) <= 1.0, "direct evaluation on the reference axis"


def test_reference_velocities_round_trip():
    """lambda_j exp(v_j / c) must equal lambda_ref exp(v_ref / c) for the returned v_j."""
    LB = LineBundle(["HI-LyA", "SiIII-1206", "CIV-1548"], ref_line_id="HI-LyA")
    v_ref = jnp.linspace(-500.0, 3500.0, 9)
    vj = np.asarray(velocities_in_line_frames(v_ref, LB, reference_line_idx=0))
    lam = np.asarray(LB.lambda0_A)
    lam_obs = lam[:, None] * np.exp(vj / phys.c_kms)
    assert np.allclose(lam_obs, lam[0] * np.exp(np.asarray(v_ref) / phys.c_kms), rtol=1e-12, atol=0)
