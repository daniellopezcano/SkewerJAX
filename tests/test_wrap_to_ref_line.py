"""Sign of the reference-line shift (``wrap_to_ref_line``, ``reference_velocities_to_line_centered``).

Run as ``pytest`` or ``python test_wrap_to_ref_line.py [--proposal]``. Without the flag the installed functions are
tested (the current repo FAILs, the approved proposal PASSes); with it the proposal of this stem is loaded first.
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__" and "--proposal" in sys.argv:          # before any skewerjax import
    sys.path.insert(0, HERE); import proposal_loader; proposal_loader.load(verbose=False)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from skewerjax.constants.lines import LineBundle
from skewerjax.constants.physical_constants import phys
from skewerjax.optical_depth import line_profile as lp
from skewerjax.optical_depth.optical_depth import OpticalDepthCalculator
from skewerjax.skewers.skewers import SkewerBundle
from skewerjax.fields.fields import Fields, ScalarField, VectorField



def test_delta_function():
    """A feature at v0 in Si III's own frame must land at v0 - delta_v in the Lya frame."""
    LB = LineBundle(["HI-LyA", "SiIII-1206"], ref_line_id="HI-LyA")
    v = (np.arange(4000) + 0.5) * 1.0
    tau = np.zeros((2, 1, v.size)); tau[:, 0, 3000] = 1.0
    c = OpticalDepthCalculator(velocities_kms=jnp.asarray(v), line_bundle=LB, optical_depth=jnp.asarray(tau))
    w = np.asarray(c.wrap_to_ref_line())
    dv = float(LB.delta_v[1])
    expected = np.mod(v[3000] - dv, v.size)
    got = v[np.argmax(w[1, 0])]
    assert abs(got - expected) <= 1.0, f"Si III at {got} km/s, expected {expected:.1f} (v0 - delta_v)"


def test_physical_absorber():
    """One static absorber seen in Lya and in an H I line at 1206.5 A: computed by OpticalDepthCalculator
    in each line's own frame, then put on the Lya frame. The 1206.5 line must appear c ln(1215.67/1206.5)
    bluer, as the equality of observed wavelengths requires."""
    LB = LineBundle(["HI-LyA", "HI-LyA-1206"], ref_line_id="HI-LyA")
    a, n_pt = 0.25, 401
    pos = np.zeros((1, n_pt, 3)); pos[0, :, 0] = np.linspace(9.9, 10.1, n_pt)          # cMpc
    los = np.zeros((1, n_pt, 3)); los[..., 0] = -1.0
    sb = SkewerBundle(spatial_points=jnp.asarray(pos), a0=a, ICs_defined_at_index=0, los_directions=jnp.asarray(los),
                      manual_scale_factors=jnp.full((1, 1, n_pt), a))
    dens = np.where(np.abs(pos[..., 0] - 10.0) < 0.01, 1e-12, 0.0)
    const = lambda val: ScalarField(lambda x, val=val: jnp.full(x.shape[:-1], val))
    fields = Fields({"HI": {"temperature": const(1e4), "turbulence": const(0.0), "collisional": const(0.0),
                            "number_density": ScalarField(lambda x: jnp.asarray(dens.reshape(x.shape[:-1]))),
                            "velocity": VectorField(lambda x: jnp.zeros(x.shape[:-1] + (3,)))}})
    v = jnp.asarray((np.arange(-2000, 2000) + 0.5) * 1.0)
    calc = OpticalDepthCalculator(skewer_bundle=sb, velocities_kms=v, line_bundle=LB, fields=fields, a_dagger=a)
    t = np.asarray(calc.compute())
    v_ly, v_own = float(v[np.argmax(t[0, 0])]), float(v[np.argmax(t[1, 0])])
    w = np.asarray(calc.wrap_to_ref_line())
    v_ref = float(v[np.argmax(w[1, 0])])
    expected = v_ly + phys.c_kms * np.log(1206.5 / 1215.6701)
    assert abs(v_own - v_ly) <= 1.0, "both lines of the same gas peak at the same line-centred velocity"
    assert abs(np.mod(v_ref - expected + 2000, 4000) - 2000) <= 1.0, f"1206 line at {v_ref} km/s on the Lya frame, expected {expected:.1f}"


def test_reference_velocities_round_trip():
    """lambda_j exp(v_j / c) must equal lambda_ref exp(v_ref / c) for the returned v_j."""
    LB = LineBundle(["HI-LyA", "SiIII-1206", "CIV-1548"], ref_line_id="HI-LyA")
    v_ref = jnp.linspace(-500.0, 3500.0, 9)
    vj = np.asarray(lp.reference_velocities_to_line_centered(v_ref, LB, reference_line_idx=0))
    lam = np.asarray(LB.lambda0_A)
    lam_obs = lam[:, None] * np.exp(vj / phys.c_kms)
    assert np.allclose(lam_obs, lam[0] * np.exp(np.asarray(v_ref) / phys.c_kms), rtol=1e-12, atol=0)


def effect_on_box_p1d(n_sk=400, seed=1):
    """Toy ensemble: Lya and a narrow metal line from the same absorbers in a periodic 4000 km/s box.
    Returns the P1D ratio (wrong sign / right sign) averaged over skewers, and its scatter for one skewer."""
    rng = np.random.default_rng(seed)
    nv, dv = 4000, 1.0
    v = (np.arange(nv) + 0.5) * dv
    dV = phys.c_kms * np.log(1215.6701 / 1206.5)
    ratios = []
    P = {+1: np.zeros(nv // 2 + 1), -1: np.zeros(nv // 2 + 1)}
    for _ in range(n_sk):
        c0 = rng.uniform(0, nv * dv, 80); t0 = rng.lognormal(-1.0, 1.2, 80)
        d = (v[None, :] - c0[:, None] + nv * dv / 2) % (nv * dv) - nv * dv / 2
        ly = (t0[:, None] * np.exp(-(d / 20.0) ** 2)).sum(0)
        me = (0.2 * t0[:, None] * np.exp(-(d / 5.0) ** 2)).sum(0)
        for sgn in (+1, -1):
            shifted = np.interp(np.mod(v + sgn * dV, nv * dv), v, me, period=nv * dv)       # sgn=+1: correct
            F = np.exp(-ly - shifted); delta = F / F.mean() - 1
            pk = dv / nv * np.abs(np.fft.rfft(delta)) ** 2
            P[sgn] += pk
            if sgn == -1:
                ratios.append(pk[1:200] / P_last[1:200])
            else:
                P_last = pk
    mean_ratio = P[-1][1:200] / P[+1][1:200]
    one = np.array(ratios)
    return float(np.max(np.abs(mean_ratio - 1))), float(np.median(np.abs(one - 1)))


if __name__ == "__main__":
    proposal = "--proposal" in sys.argv
    print(f"testing {'the proposed' if proposal else 'the current'} functions")
    for t in (test_delta_function, test_physical_absorber, test_reference_velocities_round_trip):
        try:
            t(); print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
    print("box P1D, wrong vs right sign (toy, k < 0.31 s/km): the ensemble difference is sample variance of the")
    print("Lya-metal cross term (it falls as 1/sqrt(N_sk)); every single spectrum differs:")
    for n in (100, 400, 1600):
        m, s1 = effect_on_box_p1d(n_sk=n, seed=2)
        print(f"  {n:5d} skewers: max over k of |<P_wrong>/<P_right> - 1| = {m:.3f}; one skewer: median |ratio - 1| = {s1:.2f}")
