#!/usr/bin/env python
"""Line-centred velocity frames: the sign of ``wrap_to_ref_line`` and ``velocities_in_line_frames``, and the direct
evaluation on the reference axis (``compute_optical_depth(line_frame="reference")``) (review notes, relative
line-centred velocity units). Ports ``tests/test_wrap_to_ref_line.py``. A change of frame is a change of axis: the
shift wraps only for a spectrum computed with ``velocity_period``."""
from _common import Checks

C = Checks("check_line_frames", __doc__)
import numpy as np                                                        # noqa: E402
import jax.numpy as jnp                                                   # noqa: E402
from skewerjax.core import (LineBundle, SkewerBundle, cells_from_values, compute_optical_depth, phys,  # noqa: E402
                            velocities_in_line_frames, wrap_to_ref_line)

# 1. a feature at v0 in Si III's own frame lands at v0 - delta_v on the Lya frame
LB = LineBundle(["HI-LyA", "SiIII-1206"], ref_line_id="HI-LyA")
v = (np.arange(4000) + 0.5) * 1.0
tau = np.zeros((2, 1, v.size)); tau[:, 0, 3000] = 1.0
w = np.asarray(wrap_to_ref_line(tau, v, LB))
expected = v[3000] - float(LB.delta_v[1])
C.check("delta function: Si III at v0 - delta_v on the Lya frame, no wrap (|offset| [km/s])", abs(v[np.nanargmax(w[1, 0])] - expected), 1.0)
C.check("no wrap: tau unknown (NaN) where v + delta_v leaves the grid (count of finite values there)",
        float(np.isfinite(w[1, 0][v + float(LB.delta_v[1]) > v[-1]]).sum()), 0)
wp = np.asarray(wrap_to_ref_line(np.roll(tau, -2900, axis=-1), v, LB, velocity_period=v.size * 1.0))
C.check("velocity_period = N_v dv: the shift wraps (|offset| [km/s])",
        abs(v[np.argmax(wp[1, 0])] - np.mod(v[100] - float(LB.delta_v[1]), v.size)), 1.0)

# 2. one static absorber seen in Lya and in an H I line at 1206.5 A, each in its own frame, then on the Lya frame
LB2 = LineBundle(["HI-LyA", "HI-LyA-1206"], ref_line_id="HI-LyA")
a, n_pt = 0.25, 401
pos = np.zeros((1, n_pt, 3)); pos[0, :, 0] = np.linspace(9.9, 10.1, n_pt)
los = np.zeros((1, n_pt, 3)); los[..., 0] = -1.0
sb = SkewerBundle(spatial_points=jnp.asarray(pos), a0=a, ICs_defined_at_index=0, los_directions=jnp.asarray(los),
                  manual_scale_factors=jnp.full((1, 1, n_pt), a))
dens = np.where(np.abs(pos[..., 0] - 10.0) < 0.01, 1e-12, 0.0)
# cells at the sampling points (the Voronoi cell of a point of the straight skewer is its dual segment)
fields = cells_from_values(pos[0], None, {"HI": dens[0]}, temperature=1e4, volumes=np.full(n_pt, (0.2 / (n_pt - 1)) ** 3))
vv = jnp.asarray((np.arange(-3000, 1000) + 0.5) * 1.0)          # wide enough for the shift: no wrap needed
t = np.asarray(compute_optical_depth(sb, vv, LB2, fields, a_dagger=a))
v_ly, v_own = float(vv[np.argmax(t[0, 0])]), float(vv[np.argmax(t[1, 0])])
v_ref = float(vv[np.nanargmax(np.asarray(wrap_to_ref_line(t, vv, LB2))[1, 0])])
pred = v_ly + phys.c_kms * np.log(1206.5 / 1215.6701)
C.check("same gas, two lines: same line-centred velocity in their own frames [km/s]", abs(v_own - v_ly), 1.0)
C.check("same gas: the 1206.5 line c ln(1206.5 / 1215.67) bluer on the Lya frame [km/s]", abs(v_ref - pred), 1.0)
t_ref = np.asarray(compute_optical_depth(sb, vv, LB2, fields, a_dagger=a, line_frame="reference"))
C.check("line_frame='reference': the 1206.5 line evaluated directly on the Lya axis peaks there too [km/s]",
        abs(float(vv[np.argmax(t_ref[1, 0])]) - pred), 1.0)
w2 = np.asarray(wrap_to_ref_line(t, vv, LB2, fill_value=0.0))
C.check("line_frame='reference' == wrap_to_ref_line of the own-frame tau (1 km/s grid, max |dtau| / max tau)",
        np.abs(t_ref[1] - w2[1]).max() / t_ref[1].max(), 2e-2)
C.check("line_frame='reference' leaves the reference line unchanged (max |dtau|)", np.abs(t_ref[0] - t[0]).max(), 1e-12)

# 3. lambda_j exp(v_j / c) = lambda_ref exp(v_ref / c)
LB3 = LineBundle(["HI-LyA", "SiIII-1206", "CIV-1548"], ref_line_id="HI-LyA")
v_r = jnp.linspace(-500.0, 3500.0, 9)
vj = np.asarray(velocities_in_line_frames(v_r, LB3, reference_line_idx=0))
lam = np.asarray(LB3.lambda0_A)
C.check("velocities_in_line_frames: observed wavelengths agree (max rel)",
        np.max(np.abs(lam[:, None] * np.exp(vj / phys.c_kms) / (lam[0] * np.exp(np.asarray(v_r) / phys.c_kms)) - 1)), 1e-12)
C.check("LineBundle.delta_v = c ln(lambda_ref / lambda_j) (max |diff| [km/s])",
        np.max(np.abs(np.asarray(LB3.delta_v) - phys.c_kms * np.log(lam[0] / lam))), 1e-9)
C.done()
