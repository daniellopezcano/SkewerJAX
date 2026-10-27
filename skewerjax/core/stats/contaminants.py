"""P1D metal-contamination models (McDonald et al. 2006; Ma et al.) and their fits to P_tot / P_Lya."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import differential_evolution, lsq_linear, minimize, minimize_scalar


def _xp(*args):
    """jax.numpy for JAX arrays (jit / vmap / grad), NumPy otherwise (the fits run in float64 NumPy)."""
    return jnp if any(isinstance(a, jax.Array) for a in args) else np


def _safe_exp(x, clip=80.0):
    xp = _xp(x)
    return xp.exp(xp.clip(x, -clip, clip))


def _validate_fit_mode(fit_mode):
    allowed = {"ratio", "power", "relative_power"}
    if fit_mode not in allowed:
        raise ValueError(f"fit_mode must be one of {sorted(allowed)}, got {fit_mode!r}.")


def _get_fit_mask(k, p_tot, p_lya, k_range, fit_mode="ratio"):
    """Finite, P_Lya > 0 modes with k_min < k < k_max (and P_tot != 0 for ``relative_power``)."""
    _validate_fit_mode(fit_mode)
    k, p_tot, p_lya = np.asarray(k, dtype=float), np.asarray(p_tot).real, np.asarray(p_lya).real
    mask = np.isfinite(k) & np.isfinite(p_tot) & np.isfinite(p_lya) & (p_lya > 0) & (k > k_range[0]) & (k < k_range[1])
    if fit_mode == "relative_power":
        mask = mask & (p_tot != 0)
    return mask


def _compute_residual_from_ratio(ratio_model, p_tot_fit, p_lya_fit, fit_mode="ratio"):
    """Residual of P_tot / P_Lya ("ratio"), of P_tot ("power") or of P_tot relative to itself ("relative_power")."""
    _validate_fit_mode(fit_mode)
    if fit_mode == "ratio":
        return p_tot_fit / p_lya_fit - ratio_model
    p_model = ratio_model * p_lya_fit
    if fit_mode == "power":
        return p_tot_fit - p_model
    return (p_tot_fit - p_model) / p_tot_fit


def _chi2_from_residual(residual):
    return np.sum(np.asarray(residual) ** 2)


def mcdonald_factor(k, a, delta_v=2271.0):
    """P_tot / P_Lya = 1 + a^2 + 2 a cos(k Delta_v) (McDonald et al. 2006, Si III). k [s/km], Delta_v [km/s]."""
    xp = _xp(k, a)
    k = xp.asarray(k, dtype=float)
    return 1.0 + a**2 + 2.0 * a * xp.cos(k * delta_v)


def ma_factor(k, a_auto, a_cross, k_auto, k_cross, delta_v=2271.0):
    """P_tot / P_Lya = 1 + a_auto exp(k / k_auto) + a_cross cos(k Delta_v) exp(k / k_cross) (Ma et al.).

    k_auto > 0 or inf, k_cross != 0 (typically < 0) or -inf (the constant-amplitude limit: exp(0) = 1).
    """
    xp = _xp(k, a_auto, a_cross, k_auto, k_cross)
    k = xp.asarray(k, dtype=float)
    if xp is np:
        if np.isfinite(k_auto) and k_auto <= 0:
            raise ValueError("k_auto must be positive or np.inf.")
        if np.isfinite(k_cross) and k_cross == 0:
            raise ValueError("k_cross must be non-zero or -np.inf.")
    auto_exp = _safe_exp(k / k_auto) if xp is jnp or np.isfinite(k_auto) else np.ones_like(k, dtype=float)
    cross_exp = _safe_exp(k / k_cross) if xp is jnp or np.isfinite(k_cross) else np.ones_like(k, dtype=float)
    return 1.0 + a_auto * auto_exp + a_cross * xp.cos(k * delta_v) * cross_exp


def fit_mcdonald_a(
    k,
    p_tot,
    p_lya,
    delta_v=2271.0,
    a_bounds=(-0.5, 0.5),
    k_range=(0.0, 0.03),
    fit_mode="ratio",
    optimizer_options=None,
):
    """
    Fit the McDonald amplitude a.
    """

    _validate_fit_mode(fit_mode)

    if optimizer_options is None:
        optimizer_options = {
            "xatol": 1.0e-12,
            "maxiter": 1000,
        }

    k = np.asarray(k, dtype=float)
    p_tot = np.asarray(p_tot).real
    p_lya = np.asarray(p_lya).real

    mask = _get_fit_mask(
        k=k,
        p_tot=p_tot,
        p_lya=p_lya,
        k_range=k_range,
        fit_mode=fit_mode,
    )

    k_fit = k[mask]
    p_tot_fit = p_tot[mask]
    p_lya_fit = p_lya[mask]

    if k_fit.size == 0:
        raise ValueError("No valid k modes remain after applying the fit mask.")

    def chi2(a):
        ratio_model = mcdonald_factor(
            k=k_fit,
            a=a,
            delta_v=delta_v,
        )

        residual = _compute_residual_from_ratio(
            ratio_model=ratio_model,
            p_tot_fit=p_tot_fit,
            p_lya_fit=p_lya_fit,
            fit_mode=fit_mode,
        )

        return _chi2_from_residual(residual)

    result = minimize_scalar(
        chi2,
        bounds=a_bounds,
        method="bounded",
        options=optimizer_options,
    )

    return {
        "a": result.x,
        "chi2": result.fun,
        "n_fit": k_fit.size,
        "dof": k_fit.size - 1,
        "fit_mode": fit_mode,
        "k_range": k_range,
        "delta_v": delta_v,
        "result": result,
    }


def fit_ma_model_profiled(
    k,
    p_tot,
    p_lya,
    delta_v=2271.0,
    k_range=(0.0, 0.1),
    fit_mode="ratio",
    lambda_bounds=None,
    amplitude_bounds=None,
    enforce_positive_model=False,
    linear_lsq_tol=1.0e-14,
    run_global=True,
    global_optimizer_options=None,
    polish_optimizer_options=None,
    extra_starts=None,
):
    """
    Robust fit of the Ma et al. model using inverse scales and
    profiled amplitudes.

    Instead of fitting

        a_auto, a_cross, k_auto, k_cross

    directly, we fit

        lambda_auto      = 1 / k_auto,
        lambda_cross_abs = 1 / |k_cross|,

    and solve a_auto and a_cross analytically by bounded linear
    least squares for every pair of nonlinear scale parameters.

    This is much more stable than a direct 4D L-BFGS-B fit, and it
    includes the McDonald-like constant-amplitude limit exactly at

        lambda_auto = 0,
        lambda_cross_abs = 0.
    """

    _validate_fit_mode(fit_mode)

    k = np.asarray(k, dtype=float)
    p_tot = np.asarray(p_tot).real
    p_lya = np.asarray(p_lya).real

    mask = _get_fit_mask(
        k=k,
        p_tot=p_tot,
        p_lya=p_lya,
        k_range=k_range,
        fit_mode=fit_mode,
    )

    k_fit = k[mask]
    p_tot_fit = p_tot[mask]
    p_lya_fit = p_lya[mask]

    if k_fit.size == 0:
        raise ValueError("No valid k modes remain after applying the fit mask.")

    if lambda_bounds is None:
        lambda_bounds = {
            "lambda_auto": (0.0, 500.0),
            "lambda_cross_abs": (0.0, 500.0),
        }

    if amplitude_bounds is None:
        amplitude_bounds = {
            "a_auto": (0.0, 100.0),
            "a_cross": (-100.0, 100.0),
        }

    if global_optimizer_options is None:
        global_optimizer_options = {
            "tol": 1.0e-10,
            "atol": 1.0e-12,
            "maxiter": 1000,
            "popsize": 25,
            "polish": False,
            "seed": 12345,
            "workers": 1,
            "updating": "immediate",
        }

    if polish_optimizer_options is None:
        polish_optimizer_options = {
            "maxiter": 50000,
            "ftol": 1.0e-15,
            "gtol": 1.0e-13,
            "maxls": 100,
        }

    lambda_opt_bounds = [
        lambda_bounds["lambda_auto"],
        lambda_bounds["lambda_cross_abs"],
    ]

    amp_lower = np.asarray(
        [
            amplitude_bounds["a_auto"][0],
            amplitude_bounds["a_cross"][0],
        ],
        dtype=float,
    )

    amp_upper = np.asarray(
        [
            amplitude_bounds["a_auto"][1],
            amplitude_bounds["a_cross"][1],
        ],
        dtype=float,
    )

    cos_term = np.cos(k_fit * delta_v)

    def build_linear_system(lambda_auto, lambda_cross_abs):
        """
        Build y = X @ [a_auto, a_cross] for fixed inverse scales.
        """

        basis_auto = _safe_exp(lambda_auto * k_fit)
        basis_cross = cos_term * _safe_exp(-lambda_cross_abs * k_fit)

        if fit_mode == "ratio":
            y = p_tot_fit / p_lya_fit - 1.0
            X = np.column_stack(
                [
                    basis_auto,
                    basis_cross,
                ]
            )

        elif fit_mode == "power":
            y = p_tot_fit - p_lya_fit
            X = np.column_stack(
                [
                    p_lya_fit * basis_auto,
                    p_lya_fit * basis_cross,
                ]
            )

        elif fit_mode == "relative_power":
            y = (p_tot_fit - p_lya_fit) / p_tot_fit
            X = np.column_stack(
                [
                    (p_lya_fit / p_tot_fit) * basis_auto,
                    (p_lya_fit / p_tot_fit) * basis_cross,
                ]
            )

        return X, y, basis_auto, basis_cross

    def solve_amplitudes(lambda_auto, lambda_cross_abs):
        """
        Solve a_auto and a_cross for fixed nonlinear inverse scales.
        """

        X, y, basis_auto, basis_cross = build_linear_system(
            lambda_auto=lambda_auto,
            lambda_cross_abs=lambda_cross_abs,
        )

        lsq = lsq_linear(
            X,
            y,
            bounds=(amp_lower, amp_upper),
            tol=linear_lsq_tol,
            max_iter=10000,
            lsmr_tol="auto",
        )

        a_auto, a_cross = lsq.x

        model_ratio = 1.0 + a_auto * basis_auto + a_cross * basis_cross

        residual = _compute_residual_from_ratio(
            ratio_model=model_ratio,
            p_tot_fit=p_tot_fit,
            p_lya_fit=p_lya_fit,
            fit_mode=fit_mode,
        )

        chi2 = _chi2_from_residual(residual)

        if enforce_positive_model and np.any(model_ratio <= 0.0):
            negative_part = np.minimum(model_ratio, 0.0)
            chi2 += 1.0e50 * np.sum(negative_part**2)

        return {
            "chi2": chi2,
            "a_auto": a_auto,
            "a_cross": a_cross,
            "model_ratio": model_ratio,
            "residual": residual,
            "lsq_result": lsq,
        }

    def objective(theta):
        lambda_auto, lambda_cross_abs = theta

        if lambda_auto < 0.0 or lambda_cross_abs < 0.0:
            return 1.0e100

        solved = solve_amplitudes(
            lambda_auto=lambda_auto,
            lambda_cross_abs=lambda_cross_abs,
        )

        return solved["chi2"]

    def make_candidate(theta, label, optimizer_result=None):
        theta = np.asarray(theta, dtype=float)

        theta[0] = np.clip(theta[0], *lambda_opt_bounds[0])
        theta[1] = np.clip(theta[1], *lambda_opt_bounds[1])

        solved = solve_amplitudes(
            lambda_auto=theta[0],
            lambda_cross_abs=theta[1],
        )

        return {
            "label": label,
            "theta": theta,
            "chi2": solved["chi2"],
            "a_auto": solved["a_auto"],
            "a_cross": solved["a_cross"],
            "optimizer_result": optimizer_result,
            "lsq_result": solved["lsq_result"],
        }

    candidates = []

    # ---------------------------------------------------------------
    # Candidate 1: exact constant-amplitude limit.
    # This should already be able to reproduce a McDonald-like shape.
    # ---------------------------------------------------------------

    candidates.append(
        make_candidate(
            theta=np.asarray([0.0, 0.0], dtype=float),
            label="constant_limit",
            optimizer_result=None,
        )
    )

    # ---------------------------------------------------------------
    # Candidate 2: local starts from physically motivated scales.
    # ---------------------------------------------------------------

    starts = [
        [0.0, 0.0],
        [1.0 / 0.04, 1.0 / 0.02],
        [1.0 / 0.06, 1.0 / 0.03],
        [1.0 / 0.02, 1.0 / 0.01],
        [10.0, 10.0],
        [25.0, 50.0],
        [50.0, 50.0],
        [100.0, 100.0],
    ]

    if extra_starts is not None:
        starts.extend(extra_starts)

    for theta0 in starts:
        theta0 = np.asarray(theta0, dtype=float)
        theta0[0] = np.clip(theta0[0], *lambda_opt_bounds[0])
        theta0[1] = np.clip(theta0[1], *lambda_opt_bounds[1])

        result_local = minimize(
            objective,
            theta0,
            method="L-BFGS-B",
            bounds=lambda_opt_bounds,
            options=polish_optimizer_options,
        )

        candidates.append(
            make_candidate(
                theta=result_local.x,
                label="local_start",
                optimizer_result=result_local,
            )
        )

    # ---------------------------------------------------------------
    # Candidate 3: optional global search + local polish.
    # ---------------------------------------------------------------

    if run_global:
        result_de = differential_evolution(
            objective,
            bounds=lambda_opt_bounds,
            **global_optimizer_options,
        )

        candidates.append(
            make_candidate(
                theta=result_de.x,
                label="differential_evolution",
                optimizer_result=result_de,
            )
        )

        result_polish = minimize(
            objective,
            result_de.x,
            method="L-BFGS-B",
            bounds=lambda_opt_bounds,
            options=polish_optimizer_options,
        )

        candidates.append(
            make_candidate(
                theta=result_polish.x,
                label="polished_differential_evolution",
                optimizer_result=result_polish,
            )
        )

    # ---------------------------------------------------------------
    # Select best candidate and reconstruct Ma parameters.
    # ---------------------------------------------------------------

    best_candidate = min(candidates, key=lambda item: item["chi2"])

    lambda_auto_best, lambda_cross_abs_best = best_candidate["theta"]

    solved_best = solve_amplitudes(
        lambda_auto=lambda_auto_best,
        lambda_cross_abs=lambda_cross_abs_best,
    )

    a_auto_best = solved_best["a_auto"]
    a_cross_best = solved_best["a_cross"]

    if lambda_auto_best == 0.0:
        k_auto_best = np.inf
    else:
        k_auto_best = 1.0 / lambda_auto_best

    if lambda_cross_abs_best == 0.0:
        k_cross_best = -np.inf
    else:
        k_cross_best = -1.0 / lambda_cross_abs_best

    return {
        "a_auto": a_auto_best,
        "a_cross": a_cross_best,
        "k_auto": k_auto_best,
        "k_cross": k_cross_best,
        "lambda_auto": lambda_auto_best,
        "lambda_cross_abs": lambda_cross_abs_best,
        "chi2": solved_best["chi2"],
        "n_fit": k_fit.size,
        "dof": k_fit.size - 4,
        "fit_mode": fit_mode,
        "k_range": k_range,
        "delta_v": delta_v,
        "model_ratio_fit": solved_best["model_ratio"],
        "residual_fit": solved_best["residual"],
        "lsq_result": solved_best["lsq_result"],
        "best_candidate": best_candidate,
        "all_candidates": candidates,
    }
