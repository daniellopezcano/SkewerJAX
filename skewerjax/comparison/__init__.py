"""Block C, comparison with fake_spectra: what remains is the benchmark of the two codes (wall time, memory, agreement;
``comparison.benchmark``; the scaling of notebooks 13 and 14, ``comparison.scaling``).

Nothing here re-implements fake_spectra any more: its choices are options of the functions they belong to
(``simulations.gas_to_cells``, ``core.cell_pieces``, ``core.compute_optical_depth``), and notebook 12 compares them, one
at a time, with fake_spectra itself. Depends on ``skewerjax.simulations`` and ``skewerjax.core``; nothing depends on it.
"""

__all__ = []
