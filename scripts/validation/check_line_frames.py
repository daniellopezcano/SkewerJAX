#!/usr/bin/env python
"""Line-centred velocity frames: the sign of ``wrap_to_ref_line`` and ``velocities_in_line_frames`` (review
notes, relative line-centred velocity units). Ports ``tests/test_wrap_to_ref_line.py``."""
from _common import Checks

C = Checks("check_line_frames", __doc__)
import numpy as np                                                        # noqa: E402
import jax.numpy as jnp                                                   # noqa: E402
from skewerjax.core import (Constant, Fields, LineBundle, SampledField, ScalarField, SkewerBundle, VectorField,
                            compute_optical_depth, phys, velocities_in_line_frames, wrap_to_ref_line)  # noqa: E402

# 1. a feature at v0 in Si III's own frame lands at v0 - delta_v on the Lya frame
LB = LineBundle(["HI-LyA", "SiIII-1206"], ref_line_id="HI-LyA")
v = (np.arange(4000) + 0.5) * 1.0
tau = np.zeros((2, 1, v.size)); tau[:, 0, 3000] = 1.0
w = np.asarray(wrap_to_ref_line(tau, v, LB))
expected = np.mod(v[3000] - float(LB.delta_v[1]), v.size)
C.check("delta function: Si III at v0 - delta_v on the Lya frame (|offset| [km/s])", abs(v[np.argmax(w[1, 0])] - expected), 1.0)

# 2. one static absorber seen in Lya and in an H I line at 1206.5 A, each in its own frame, then on the Lya frame
LB2 = LineBundle(["HI-LyA", "HI-LyA-1206"], ref_line_id="HI-LyA")
a, n_pt = 0.25, 401
pos = np.zeros((1, n_pt, 3)); pos[0, :, 0] = np.linspace(9.9, 10.1, n_pt)
los = np.zeros((1, n_pt, 3)); los[..., 0] = -1.0
sb = SkewerBundle(spatial_points=jnp.asarray(pos), a0=a, ICs_defined_at_index=0, los_directions=jnp.asarray(los),
                  manual_scale_factors=jnp.full((1, 1, n_pt), a))
dens = np.where(np.abs(pos[..., 0] - 10.0) < 0.01, 1e-12, 0.0)
fields = Fields({"HI": {"temperature": ScalarField(Constant(1e4)), "turbulence": ScalarField(Constant(0.0)),
                        "collisional": ScalarField(Constant(0.0)), "number_density": ScalarField(SampledField(dens)),
                        "velocity": VectorField(SampledField(np.zeros((1, n_pt, 3))))}})
vv = jnp.asarray((np.arange(-2000, 2000) + 0.5) * 1.0)
t = np.asarray(compute_optical_depth(sb, vv, LB2, fields, a_dagger=a))
v_ly, v_own = float(vv[np.argmax(t[0, 0])]), float(vv[np.argmax(t[1, 0])])
v_ref = float(vv[np.argmax(np.asarray(wrap_to_ref_line(t, vv, LB2))[1, 0])])
pred = v_ly + phys.c_kms * np.log(1206.5 / 1215.6701)
C.check("same gas, two lines: same line-centred velocity in their own frames [km/s]", abs(v_own - v_ly), 1.0)
C.check("same gas: the 1206.5 line c ln(1206.5 / 1215.67) bluer on the Lya frame [km/s]",
        abs(np.mod(v_ref - pred + 2000, 4000) - 2000), 1.0)

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
