"""SkewerJAX against fake_spectra itself on TNG50-4 snapshot 25: with every option at fake_spectra's value the two codes
agree (flux and column, H I and Si III), and the default differs from it.
fake_spectra runs natively in this process (one ``Spectra``): markers ``tng``, ``fake_spectra`` and ``slow`` (minutes)."""

import tempfile
from functools import cache

import numpy as np
import pytest

from skewerjax.core import (
    LineBundle,
    SnapshotFrame,
    axis_aligned_skewer_points,
    cell_pieces,
    compute_optical_depth,
    density_per_pixel,
)
from skewerjax.simulations import ArepoHDF5Reader, gas_to_cells, select_cells_near_paths
from skewerjax.utils.paths import snapshot_dir

from helpers import SKEWER_SETS

pytestmark = [pytest.mark.tng, pytest.mark.fake_spectra, pytest.mark.slow]
TNG = snapshot_dir("TNG50-4")
LINES = LineBundle(["HI-LyA", "SiIII-1206"], ref_line_id="HI-LyA")
CELLS = dict(
    mass_unit="proton",
    T_constants="fake_spectra",
    x_HI_model="fake_sf",
    cloudy_density="fake_spectra",
    cloudy_interpolation="cubic",
)
TAU = dict(voigt="harris", detuning="linear", b_constants="fake_spectra", pixel_average=True, tautail=1e-7, piece_chunk=2048)


@cache
def setup():
    from fake_spectra.spectra import Spectra

    rd = ArepoHDF5Reader(TNG, 25)
    H = rd.header
    s = SKEWER_SETS["nset"]
    cofm, axis = s["cofm"][:6].copy(), s["axis"][:6].copy()
    with tempfile.TemporaryDirectory(prefix="fs_") as work:  # fake_spectra recomputes (reload_file); nothing kept
        fs = Spectra(25, str(TNG), cofm.copy(), axis.copy(), res=1.0, savefile="t.hdf5", savedir=work, reload_file=True, quiet=True)
        native = dict(
            HI=dict(tau=np.asarray(fs.get_tau("H", 1, 1215)), n=np.asarray(fs.get_density("H", 1))),
            SiIII=dict(tau=np.asarray(fs.get_tau("Si", 3, 1206)), n=np.asarray(fs.get_density("Si", 3))),
        )
    points = axis_aligned_skewer_points(cofm * H.code_length_cMpc, axis, H.box_cMpc, 0.25 / H.velfac_kms_per_cMpc)
    nbins, dv = H.pixel_grid(1.0)
    assert fs.nbins == nbins and abs(fs.dvbin - dv) < 1e-12 and fs.kernel_int == 0
    return rd, H, points, select_cells_near_paths(rd, points), native, (np.arange(nbins) + 0.5) * dv, dv


def matched():
    rd, H, points, gas, native, v, dv = setup()
    cells = gas_to_cells(gas, ["HI", "SiIII"], **CELLS)
    frame = SnapshotFrame(H.a, H.velfac_kms_per_cMpc, path_length=np.full(points.shape[0], H.box_cMpc))
    pieces = cell_pieces(cells, LINES, points=points, frame=frame, deposition="tophat_chords", chord_weight=1 / 8, a_data=H.a)
    return (
        np.asarray(compute_optical_depth(pieces, v, LINES, velocity_period=v.size * dv, resolution_tolerance=None, **TAU)),
        cells,
        frame,
    )


def test_the_matched_options_reproduce_fake_spectra():
    _, _, _, _, native, _, _ = setup()
    tau, _, _ = matched()
    for j, (sp, tol) in enumerate((("HI", 5e-6), ("SiIII", 5e-5))):
        F, Fn = np.exp(-tau[j]), np.exp(-native[sp]["tau"])
        assert np.abs(F - Fn).max() < tol, (sp, np.abs(F - Fn).max())
        assert np.sqrt(np.mean((F - Fn) ** 2)) < 1e-6, sp


def test_the_tophat_column_is_fake_spectras():
    rd, H, points, gas, native, v, dv = setup()
    _, cells, frame = matched()
    _, parts = cell_pieces(cells, LINES, points=points, frame=frame, deposition="tophat", a_data=H.a, return_parts=True)
    for sp in ("HI", "SiIII"):
        n, nn = density_per_pixel(parts["absorbers"][sp], H.box_cMpc, v.size), native[sp]["n"]
        has = nn > 0
        assert np.median(np.abs(n[has] / nn[has] - 1)) < 1e-5, sp
        assert np.abs(n.sum() / nn.sum() - 1) < 1e-4, sp


def test_the_default_differs():
    rd, H, points, gas, native, v, dv = setup()
    frame = SnapshotFrame(H.a, H.velfac_kms_per_cMpc, path_length=np.full(points.shape[0], H.box_cMpc))
    cells0 = gas_to_cells(gas, ["HI", "SiIII"])
    tau_def = np.asarray(
        compute_optical_depth(
            cell_pieces(cells0, LINES, points=points, frame=frame),
            v,
            LINES,
            velocity_period=v.size * dv,
            resolution_tolerance=None,
        )
    )
    rms = np.sqrt(np.mean((np.exp(-tau_def[0]) - np.exp(-native["HI"]["tau"])) ** 2))
    assert rms > 1e-3, rms  # the default is another spectrum
    tau_m, _, _ = matched()
    assert np.sqrt(np.mean((np.exp(-tau_def[0]) - np.exp(-tau_m[0])) ** 2)) > 1e-3


def test_the_neutral_fraction_rule_through_star_forming_cells():
    """``x_HI_model="fake_sf"`` against fake_spectra's ``sf_neutral=True``, on a skewer through the densest cell of the first
    block of the snapshot: it holds hundreds of star-forming cells, which the 6 skewers above do not."""
    from fake_spectra.spectra import Spectra

    rd, H, *_ = setup()
    _, _, blk = next(iter(rd.iter_blocks(("Coordinates", "Density"), max_rows=4_000_000)))
    centre = np.asarray(blk["Coordinates"][int(np.argmax(blk["Density"]))], dtype=float)
    cofm, axis = np.array([[centre[0], 0.0, centre[2]]]), np.array([2], dtype=np.int32)
    with tempfile.TemporaryDirectory(prefix="fs_") as work:
        fs = Spectra(25, str(TNG), cofm.copy(), axis.copy(), res=1.0, savefile="s.hdf5", savedir=work, reload_file=True, quiet=True)
        native = np.asarray(fs.get_density("H", 1))[0]
    points = axis_aligned_skewer_points(cofm * H.code_length_cMpc, axis, H.box_cMpc, 0.25 / H.velfac_kms_per_cMpc)
    gas = select_cells_near_paths(rd, points)
    assert (np.asarray(gas.fields["sfr"]) > 0).sum() > 100
    frame = SnapshotFrame(H.a, H.velfac_kms_per_cMpc, path_length=np.array([H.box_cMpc]))
    has = native > 0

    def rms(model):
        cells = gas_to_cells(gas, "HI", mass_unit="proton", T_constants="fake_spectra", x_HI_model=model)
        lya = LineBundle(["HI-LyA"], ref_line_id="HI-LyA")
        _, parts = cell_pieces(cells, lya, points=points, frame=frame, deposition="tophat", a_data=H.a, return_parts=True)
        n = density_per_pixel(parts["absorbers"]["HI"], H.box_cMpc, native.size)[0]
        return np.sqrt(np.mean((n[has] / native[has] - 1) ** 2))

    assert rms("fake_sf") < 1e-5, rms("fake_sf")
    assert rms("stored_sf") > 1e-4, rms("stored_sf")  # the default is another rule on these cells
