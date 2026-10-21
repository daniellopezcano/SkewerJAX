"""Block C, comparison with fake_spectra: its per-cell scheme on SkewerJAX absorbers, the matched options, the effect
chain, and fake_spectra's own results (live or from a reference file).

Depends on ``skewerjax.simulations`` and ``skewerjax.core``; nothing depends on it. The default options everywhere are
SkewerJAX's; every "matched" difference is one named switch (``effects.EXTRACTION_EFFECTS``, ``INTEGRATION_EFFECTS``).
"""
from skewerjax.comparison.emulation import (MatchedIntegration, apply_tautail, fake_chord_absorbers, fake_emulated_tau,
                                            fake_pixel_points, matched_tau)
from skewerjax.comparison.effects import (DEFAULT_INTEGRATION, EXTRACTION_EFFECTS, FAKE_SPECTRA_EXTRACTION, FAKE_SPECTRA_INTEGRATION,
                                          INTEGRATION_EFFECTS, Effect, effect_chain, effect_rows, print_effect_rows, run_effects,
                                          spectral_metrics, velocity_lag_kms)
from skewerjax.comparison.source import SKEWER_SETS, FakeSpectraSource

__all__ = [n for n in dir() if not n.startswith("_")]
