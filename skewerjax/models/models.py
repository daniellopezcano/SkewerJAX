# models.py
# McDonald et al. correlated-metal shifted-copy model
#
# Field-level ansatz:
#
#   delta_total(v) = delta_Lya(v) + sum_m A_m delta_Lya(v + Delta_v_m)
#
# This script returns the individual modeled line fields:
#
#   overflux_mcdonald.shape = (N_lines, N_skewer, N_v)
#
# where the Lyalpha/reference entry is the input overflux_lya and each
# metal entry is A_m times a periodically shifted copy of overflux_lya.

import jax.numpy as jnp


def _effective_uniform_dv(
    velocities_kms,
    *,
    uniformity_rtol=1e-3,
    uniformity_atol_kms=None,
    atol_eps_factor=64.0,
):
    """
    Estimate an effective velocity spacing for an almost-uniform grid.

    This is deliberately tolerant to small residuals from jnp.linspace,
    especially when JAX is running in default float32 mode.

    Parameters
    ----------
    velocities_kms : array, shape (N_v,)
        Velocity grid in km/s.
    uniformity_rtol : float
        Relative tolerance with respect to |dv|.
    uniformity_atol_kms : float or None
        Absolute tolerance in km/s. If None, a dtype-aware value is used.
    atol_eps_factor : float
        Multiplier for machine-epsilon-based absolute tolerance.

    Returns
    -------
    dv : scalar
        Effective velocity spacing, estimated as median(diff(velocities_kms)).
    """
    v = jnp.asarray(velocities_kms)

    if v.ndim != 1:
        raise ValueError(f"`velocities_kms` must be 1D. Got shape {v.shape}.")

    if v.shape[0] < 2:
        raise ValueError("`velocities_kms` must contain at least two points.")

    if not jnp.issubdtype(v.dtype, jnp.floating):
        raise TypeError("`velocities_kms` must have a floating dtype.")

    dv_all = jnp.diff(v)
    dv = jnp.median(dv_all)

    if uniformity_atol_kms is None:
        eps = jnp.finfo(v.dtype).eps
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(v)))
        uniformity_atol_kms = atol_eps_factor * eps * scale

    max_abs_dev = jnp.max(jnp.abs(dv_all - dv))
    threshold = uniformity_atol_kms + uniformity_rtol * jnp.abs(dv)

    if not bool(max_abs_dev <= threshold):
        raise ValueError(
            "`velocities_kms` is not uniformly spaced within tolerance.\n"
            f"  dtype                = {v.dtype}\n"
            f"  N_v                  = {v.shape[0]}\n"
            f"  dv_median            = {float(dv)} km/s\n"
            f"  min(diff)            = {float(jnp.min(dv_all))} km/s\n"
            f"  max(diff)            = {float(jnp.max(dv_all))} km/s\n"
            f"  max |diff-dv_median| = {float(max_abs_dev)} km/s\n"
            f"  tolerance            = {float(threshold)} km/s\n"
            f"  uniformity_rtol      = {uniformity_rtol}\n"
            f"  uniformity_atol_kms  = {float(uniformity_atol_kms)}\n"
        )

    return dv


def _periodic_shift_uniform_1d(
    field,
    velocities_kms,
    shift_kms,
    *,
    uniformity_rtol=1e-3,
    uniformity_atol_kms=None,
    integer_tol=1e-5,
):
    """
    Periodically evaluate field(v + shift_kms) on an almost-uniform grid.

    Parameters
    ----------
    field : array, shape (..., N_v)
        Field to shift. The velocity axis must be the last axis.
    velocities_kms : array, shape (N_v,)
        Velocity grid in km/s.
    shift_kms : float
        Shift in km/s.

        Positive shift means

            output[..., j] = field(..., v_j + shift_kms)

        with periodic wrapping.
    uniformity_rtol : float
        Relative tolerance used to accept the velocity grid as uniform.
    uniformity_atol_kms : float or None
        Absolute tolerance in km/s.
    integer_tol : float
        If shift/dv is closer than this to an integer, snap to the integer shift.

    Returns
    -------
    shifted : array, shape (..., N_v)
        Periodically shifted field.
    """
    field = jnp.asarray(field)
    velocities_kms = jnp.asarray(velocities_kms)

    if velocities_kms.ndim != 1:
        raise ValueError(
            f"`velocities_kms` must be 1D. Got shape {velocities_kms.shape}."
        )

    if field.shape[-1] != velocities_kms.shape[0]:
        raise ValueError(
            "Last axis of `field` must match len(velocities_kms). "
            f"Got field.shape[-1]={field.shape[-1]} and "
            f"len(velocities_kms)={velocities_kms.shape[0]}."
        )

    dv = _effective_uniform_dv(
        velocities_kms,
        uniformity_rtol=uniformity_rtol,
        uniformity_atol_kms=uniformity_atol_kms,
    )

    N_v = velocities_kms.shape[0]

    # Convert physical shift to pixel shift.
    s_raw = jnp.asarray(shift_kms, dtype=field.dtype) / dv

    # Avoid tiny interpolation artefacts when the shift is effectively integer.
    s_int = jnp.round(s_raw)
    s = jnp.where(jnp.abs(s_raw - s_int) < integer_tol, s_int, s_raw)

    # output(v_j) = input(v_j + shift)
    x = jnp.arange(N_v, dtype=field.dtype) + s

    i0_float = jnp.floor(x)
    frac = x - i0_float

    i0 = jnp.mod(i0_float.astype(jnp.int32), N_v)
    i1 = jnp.mod(i0 + 1, N_v)

    f0 = jnp.take(field, i0, axis=-1)
    f1 = jnp.take(field, i1, axis=-1)

    frac_shape = (1,) * (field.ndim - 1) + (N_v,)
    frac = frac.reshape(frac_shape)

    return (1.0 - frac) * f0 + frac * f1


def build_mcdonald_overflux_fields(
    overflux_lya,
    velocities_kms,
    line_bundle,
    amplitudes,
    *,
    ref_line_index=None,
    shift_sign=+1.0,
    uniformity_rtol=1e-3,
    uniformity_atol_kms=None,
    integer_tol=1e-5,
):
    """
    Build McDonald-style correlated-metal overflux fields.

    McDonald field-level model:

        delta_total(v) = delta_Lya(v)
                       + sum_m A_m delta_Lya(v + Delta_v_m)

    Parameters
    ----------
    overflux_lya : array, shape (N_skewer, N_v)
        Lyalpha-only overflux field:

            delta_Lya = F_Lya / <F_Lya> - 1

    velocities_kms : array, shape (N_v,)
        Velocity grid in km/s.

    line_bundle : object
        Must expose

            line_bundle.delta_v

        with one velocity offset per line, measured relative to the Lyalpha
        reference line.

    amplitudes : array, shape (N_lines - 1,)
        One free amplitude per non-Lyalpha line.

        Example for ["HI-LyA", "SiIII-1206"]:

            amplitudes = jnp.array([0.04])

    ref_line_index : int or None
        Index of the Lyalpha/reference line.

        If None, it is inferred as the line whose |delta_v| is smallest.

    shift_sign : float
        Sign convention for the shift.

        Use +1 for

            delta_m(v) = A_m delta_Lya(v + Delta_v_m)

        Use -1 if your line_bundle.delta_v convention is opposite.

    uniformity_rtol : float
        Relative tolerance for accepting the velocity grid as uniform.

    uniformity_atol_kms : float or None
        Absolute tolerance in km/s. If None, a dtype-aware value is used.

    integer_tol : float
        Tolerance for snapping shift/dv to an integer pixel shift.

    Returns
    -------
    overflux_mcdonald : array, shape (N_lines, N_skewer, N_v)
        Line-ordered stack of McDonald fields.

        Reference line:
            delta_Lya(v)

        Metal line m:
            A_m delta_Lya(v + Delta_v_m)
    """
    overflux_lya = jnp.asarray(overflux_lya)
    velocities_kms = jnp.asarray(velocities_kms)
    amplitudes = jnp.asarray(amplitudes)

    if overflux_lya.ndim != 2:
        raise ValueError(
            "`overflux_lya` must have shape (N_skewer, N_v). "
            f"Got shape {overflux_lya.shape}."
        )

    if velocities_kms.ndim != 1:
        raise ValueError(
            "`velocities_kms` must have shape (N_v,). "
            f"Got shape {velocities_kms.shape}."
        )

    if overflux_lya.shape[-1] != velocities_kms.shape[0]:
        raise ValueError(
            "`overflux_lya.shape[-1]` must match len(velocities_kms). "
            f"Got {overflux_lya.shape[-1]} and {velocities_kms.shape[0]}."
        )

    if not hasattr(line_bundle, "delta_v"):
        raise AttributeError("`line_bundle` must expose an attribute `delta_v`.")

    delta_v = jnp.asarray(line_bundle.delta_v, dtype=overflux_lya.dtype)

    if delta_v.ndim != 1:
        raise ValueError(
            "`line_bundle.delta_v` must be 1D, with one value per line. "
            f"Got shape {delta_v.shape}."
        )

    n_lines = int(delta_v.shape[0])

    if ref_line_index is None:
        ref_line_index = int(jnp.argmin(jnp.abs(delta_v)))

    if not (0 <= ref_line_index < n_lines):
        raise ValueError(
            f"`ref_line_index` must be between 0 and {n_lines - 1}. "
            f"Got {ref_line_index}."
        )

    metal_indices = [i for i in range(n_lines) if i != ref_line_index]

    assert amplitudes.ndim == 1, (
        "`amplitudes` must be a 1D array with one amplitude per non-Lyalpha line."
    )

    assert amplitudes.shape[0] == len(metal_indices), (
        "Number of amplitudes must match number of non-Lyalpha lines. "
        f"Got {amplitudes.shape[0]} amplitudes for "
        f"{len(metal_indices)} non-Lyalpha lines."
    )

    fields = []
    amp_counter = 0

    for i_line in range(n_lines):
        if i_line == ref_line_index:
            fields.append(overflux_lya)
        else:
            shift_i = shift_sign * delta_v[i_line]
            amp_i = amplitudes[amp_counter]

            shifted_lya = _periodic_shift_uniform_1d(
                overflux_lya,
                velocities_kms,
                shift_i,
                uniformity_rtol=uniformity_rtol,
                uniformity_atol_kms=uniformity_atol_kms,
                integer_tol=integer_tol,
            )

            fields.append(amp_i * shifted_lya)
            amp_counter += 1

    return jnp.stack(fields, axis=0)


def build_mcdonald_total_overflux(
    overflux_lya,
    velocities_kms,
    line_bundle,
    amplitudes,
    **kwargs,
):
    """
    Convenience wrapper returning the summed McDonald total overflux field.

    Returns
    -------
    overflux_total : array, shape (N_skewer, N_v)
        delta_total(v) = sum_i delta_i(v)

    overflux_lines : array, shape (N_lines, N_skewer, N_v)
        Individual modeled line fields.
    """
    overflux_lines = build_mcdonald_overflux_fields(
        overflux_lya=overflux_lya,
        velocities_kms=velocities_kms,
        line_bundle=line_bundle,
        amplitudes=amplitudes,
        **kwargs,
    )

    overflux_total = jnp.sum(overflux_lines, axis=0)

    return overflux_total, overflux_lines