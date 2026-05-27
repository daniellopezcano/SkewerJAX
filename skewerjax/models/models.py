import jax.numpy as jnp


def siiii_mcdonald2006_p1d(
    k,
    Pk_lya,
    *,
    a_siiii=0.04,
    f_siiii=None,
    Fbar=None,
    delta_v_siiii=2271.0,
    include_siiii_auto=True,
):
    """
    McDonald et al. 2006 SiIII model for correlated Lyα-SiIII contamination.

    Model:
        δ_tot(v) = δ_Lyα(v) + a δ_Lyα(v + Δv)

    Therefore:
        P_SiIII(k)       = a^2 P_Lyα(k)
        P_Lyα-SiIII(k)   = 2 a cos(k Δv) P_Lyα(k)
        P_tot(k)         = [1 + a^2 + 2a cos(k Δv)] P_Lyα(k)

    Parameters
    ----------
    k : array
        1D wavenumbers in s/km, as in your notebook stats.k.
    Pk_lya : array
        Lyα-only auto P1D evaluated on k.
    a_siiii : float or array, optional
        Direct SiIII/Lyα flux-contrast scaling amplitude.
        McDonald et al. find a ~ 0.04 depending on redshift.
    f_siiii : float, optional
        If provided with Fbar, use a = f_siiii / (1 - Fbar).
        McDonald et al. quote f ~ 0.011.
    Fbar : float or array, optional
        Mean Lyα transmission, used only if f_siiii is not None.
    delta_v_siiii : float
        Lyα-SiIII velocity separation in km/s. Default: 2271 km/s.
    include_siiii_auto : bool
        If False, drops the a^2 P_Lyα SiIII auto term and keeps only the cross term.

    Returns
    -------
    dict with P_lya, P_siiii_auto, P_lya_siiii_cross, P_siiii_contam, P_total, ratio_total_to_lya.
    """
    k = jnp.asarray(k)
    Pk_lya = jnp.asarray(Pk_lya)

    if f_siiii is not None:
        if Fbar is None:
            raise ValueError("If f_siiii is provided, Fbar must also be provided so that a = f_siiii / (1 - Fbar).")
        a = jnp.asarray(f_siiii) / (1.0 - jnp.asarray(Fbar))
    else:
        a = jnp.asarray(a_siiii)

    # Allow a(z) to broadcast over k if a is vector-like.
    if getattr(a, "ndim", 0) > 0 and k.ndim == 1:
        a = a[..., None]
        k_eval = k[None, :]
    else:
        k_eval = k

    P_siiii_auto = (a**2) * Pk_lya if include_siiii_auto else jnp.zeros_like(Pk_lya)
    P_lya_siiii_cross = 2.0 * a * jnp.cos(delta_v_siiii * k_eval) * Pk_lya

    P_siiii_contam = P_siiii_auto + P_lya_siiii_cross
    P_total = Pk_lya + P_siiii_contam

    return {
        "P_lya": Pk_lya,
        "P_siiii_auto": P_siiii_auto,
        "P_lya_siiii_cross": P_lya_siiii_cross,
        "P_siiii_contam": P_siiii_contam,
        "P_total": P_total,
        "ratio_total_to_lya": jnp.where(Pk_lya != 0, P_total / Pk_lya, jnp.nan),
    }


def siiii_ma2026_p1d(
    k,
    Pk_lya,
    z,
    *,
    method="table",
    A=1.0,
    B=1.0,
    C=1.0,
    D=1.0,
    delta_v_siiii=2271.0,
):
    """
    Ma et al. 2026 improved SiIII model for correlated Lyα-SiIII contamination.

    Model:
        P_tot / P_Lyα =
            1
            + C a_auto(z) exp[k / (A k_auto(z))]
            + D a_cross(z) exp[k / (B k_cross(z))] cos(k Δv)

    where k_cross(z) is negative, so the cross oscillations are damped at high k.

    Parameters
    ----------
    k : array
        1D wavenumbers in s/km, as in your notebook stats.k.
    Pk_lya : array
        Lyα-only auto P1D evaluated on k.
    z : float or array
        Redshift. Calibrated range in the paper: 2.2 <= z <= 5.0.
    method : {"table", "powerlaw"}
        "table"    : Method I, interpolate Table 1 fiducial best-fit values.
        "powerlaw" : Method II, use approximate redshift power laws.
    A, B, C, D : floats
        Nuisance scalings for k_auto, k_cross, a_auto, a_cross.
        Fiducial model is A=B=C=D=1.
        Suggested prior ranges in the paper:
            0.7 <= A <= 1.3
            0.7 <= B <= 1.5
            0.0 <= C <= 1.5
            0.0 <= D <= 1.3
    delta_v_siiii : float
        Lyα-SiIII velocity separation in km/s. Default: 2271 km/s.

    Returns
    -------
    dict with P_lya, P_siiii_auto, P_lya_siiii_cross, P_siiii_contam, P_total,
    ratio_total_to_lya, and the effective model parameters.
    """
    k = jnp.asarray(k)
    Pk_lya = jnp.asarray(Pk_lya)
    z = jnp.asarray(z)

    method = method.lower()

    if method in ("table", "method_i", "interp", "interpolation"):
        # Table 1 of Ma et al. 2026.
        z_grid = jnp.array(
            [2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6,
             3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0]
        )

        k_auto = 1e-2 * jnp.interp(
            z, z_grid,
            jnp.array([2.96, 3.06, 3.22, 3.48, 3.84, 4.23, 4.62, 4.99,
                       5.22, 5.30, 5.63, 5.81, 5.85, 5.90, 5.91])
        )

        k_cross = 1e-2 * jnp.interp(
            z, z_grid,
            jnp.array([-1.52, -1.52, -1.47, -1.59, -1.58, -1.77, -1.88, -2.00,
                       -2.14, -2.25, -2.25, -2.42, -2.32, -2.36, -2.33])
        )

        a_auto = 1e-3 * jnp.interp(
            z, z_grid,
            jnp.array([0.99, 1.14, 1.49, 2.13, 3.27, 4.13, 4.91, 5.61,
                       5.65, 5.44, 5.46, 5.91, 5.58, 5.30, 4.80])
        )

        a_cross = 1e-2 * jnp.interp(
            z, z_grid,
            jnp.array([1.30, 1.46, 1.91, 2.36, 2.79, 3.33, 3.66, 3.83,
                       3.64, 3.74, 3.60, 3.42, 3.31, 3.17, 3.12])
        )

    elif method in ("powerlaw", "method_ii"):
        q = (1.0 + z) / 4.0

        k_auto = 3.84e-2 * q**1.26
        k_cross = -1.58e-2 * q**1.15
        a_auto = 3.27e-3 * q**1.66
        a_cross = 2.79e-2 * q**0.79

    else:
        raise ValueError("method must be 'table'/'method_i' or 'powerlaw'/'method_ii'.")

    # Broadcast z-dependent parameters over k if z is array-like.
    if z.ndim > 0 and k.ndim == 1:
        k_eval = k[None, :]
        k_auto_eval = k_auto[..., None]
        k_cross_eval = k_cross[..., None]
        a_auto_eval = a_auto[..., None]
        a_cross_eval = a_cross[..., None]
    else:
        k_eval = k
        k_auto_eval = k_auto
        k_cross_eval = k_cross
        a_auto_eval = a_auto
        a_cross_eval = a_cross

    k_auto_eff = A * k_auto_eval
    k_cross_eff = B * k_cross_eval
    a_auto_eff = C * a_auto_eval
    a_cross_eff = D * a_cross_eval

    P_siiii_auto = a_auto_eff * jnp.exp(k_eval / k_auto_eff) * Pk_lya
    P_lya_siiii_cross = (
        a_cross_eff
        * jnp.exp(k_eval / k_cross_eff)
        * jnp.cos(delta_v_siiii * k_eval)
        * Pk_lya
    )

    P_siiii_contam = P_siiii_auto + P_lya_siiii_cross
    P_total = Pk_lya + P_siiii_contam

    return {
        "P_lya": Pk_lya,
        "P_siiii_auto": P_siiii_auto,
        "P_lya_siiii_cross": P_lya_siiii_cross,
        "P_siiii_contam": P_siiii_contam,
        "P_total": P_total,
        "ratio_total_to_lya": jnp.where(Pk_lya != 0, P_total / Pk_lya, jnp.nan),
        "k_auto": k_auto,
        "k_cross": k_cross,
        "a_auto": a_auto,
        "a_cross": a_cross,
    }