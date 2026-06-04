import numpy as np

from scipy.optimize import (
    minimize,
    minimize_scalar,
    differential_evolution,
    lsq_linear,
)


# ===============================================================
# Numerical utilities
# ===============================================================

def _safe_exp(x, clip=80.0):
    """
    Numerically safe exponential.
    """
    return np.exp(np.clip(x, -clip, clip))


def _validate_fit_mode(fit_mode):
    allowed = {"ratio", "power", "relative_power"}

    if fit_mode not in allowed:
        raise ValueError(
            f"fit_mode must be one of {sorted(allowed)}, got {fit_mode!r}."
        )


def _get_fit_mask(k, p_tot, p_lya, k_range, fit_mode="ratio"):
    """
    Build a common mask for fitting P_tot and P_Lya.
    """

    _validate_fit_mode(fit_mode)

    k = np.asarray(k, dtype=float)
    p_tot = np.asarray(p_tot).real
    p_lya = np.asarray(p_lya).real

    k_min, k_max = k_range

    mask = (
        np.isfinite(k)
        & np.isfinite(p_tot)
        & np.isfinite(p_lya)
        & (p_lya > 0)
        & (k > k_min)
        & (k < k_max)
    )

    if fit_mode == "relative_power":
        mask = mask & (p_tot != 0)

    return mask


def _compute_residual_from_ratio(
    ratio_model,
    p_tot_fit,
    p_lya_fit,
    fit_mode="ratio",
):
    """
    Compute residuals for the requested fitting mode.
    """

    _validate_fit_mode(fit_mode)

    ratio_fit = p_tot_fit / p_lya_fit

    if fit_mode == "ratio":
        residual = ratio_fit - ratio_model

    elif fit_mode == "power":
        p_model = ratio_model * p_lya_fit
        residual = p_tot_fit - p_model

    elif fit_mode == "relative_power":
        p_model = ratio_model * p_lya_fit
        residual = (p_tot_fit - p_model) / p_tot_fit

    return residual


def _chi2_from_residual(residual):
    """
    Unweighted chi-square.
    """
    return np.sum(np.asarray(residual) ** 2)


# ===============================================================
# McDonald et al. model
# ===============================================================

def mcdonald_factor(k, a, delta_v=2271.0):
    """
    Multiplicative correction P_tot / P_Lya for the McDonald model.

        P_tot(k) / P_Lya(k) = 1 + a^2 + 2 a cos(k Delta_v)

    Parameters
    ----------
    k : array
        Wavenumbers in s/km.
    a : float
        McDonald SiIII contamination amplitude.
    delta_v : float
        Ly-alpha--SiIII velocity separation in km/s.

    Returns
    -------
    factor : array
        Multiplicative correction factor.
    """

    k = np.asarray(k, dtype=float)

    return 1.0 + a**2 + 2.0 * a * np.cos(k * delta_v)


def mcdonald_p1d(k, p_lya, a, delta_v=2271.0):
    """
    Total P1D predicted by the McDonald model.
    """

    p_lya = np.asarray(p_lya).real

    return mcdonald_factor(k, a, delta_v=delta_v) * p_lya


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


def mcdonald_as_ma_parameters(a):
    """
    Return the Ma constant-limit parameters that exactly reproduce
    the McDonald factor:

        1 + a^2 + 2 a cos(k Delta_v)

    using

        1 + a_auto + a_cross cos(k Delta_v)

    with

        a_auto = a^2,
        a_cross = 2a,
        k_auto = infinity,
        k_cross = -infinity.
    """

    return {
        "a_auto": a**2,
        "a_cross": 2.0 * a,
        "k_auto": np.inf,
        "k_cross": -np.inf,
        "lambda_auto": 0.0,
        "lambda_cross_abs": 0.0,
    }


# ===============================================================
# Ma et al. model
# ===============================================================

def ma_factor(
    k,
    a_auto,
    a_cross,
    k_auto,
    k_cross,
    delta_v=2271.0,
):
    """
    Multiplicative correction P_tot / P_Lya for the Ma et al. model.

        P_tot(k) / P_Lya(k)
            = 1
            + a_auto  exp(k / k_auto)
            + a_cross cos(k Delta_v) exp(k / k_cross)

    Parameters
    ----------
    k : array
        Wavenumbers in s/km.
    a_auto : float
        Amplitude of the SiIII auto-power contribution.
    a_cross : float
        Amplitude of the Ly-alpha--SiIII cross-power contribution.
    k_auto : float
        Scale controlling the small-scale auto-power enhancement.
        Expected positive. np.inf gives the constant limit.
    k_cross : float
        Scale controlling the damping of the oscillatory cross term.
        In Ma et al., typically negative. -np.inf gives the constant limit.
    delta_v : float
        Ly-alpha--SiIII velocity separation in km/s.

    Returns
    -------
    factor : array
        Multiplicative correction factor.
    """

    k = np.asarray(k, dtype=float)

    if np.isfinite(k_auto) and k_auto <= 0:
        raise ValueError("k_auto must be positive or np.inf.")

    if np.isfinite(k_cross) and k_cross == 0:
        raise ValueError("k_cross must be non-zero or -np.inf.")

    if np.isfinite(k_auto):
        auto_exp = _safe_exp(k / k_auto)
    else:
        auto_exp = np.ones_like(k, dtype=float)

    if np.isfinite(k_cross):
        cross_exp = _safe_exp(k / k_cross)
    else:
        cross_exp = np.ones_like(k, dtype=float)

    auto_term = a_auto * auto_exp

    cross_term = (
        a_cross
        * np.cos(k * delta_v)
        * cross_exp
    )

    return 1.0 + auto_term + cross_term


def ma_factor_from_lambdas(
    k,
    a_auto,
    a_cross,
    lambda_auto,
    lambda_cross_abs,
    delta_v=2271.0,
):
    """
    Ma factor in inverse-scale variables.

        lambda_auto      = 1 / k_auto
        lambda_cross_abs = 1 / |k_cross|

    Model:

        P_tot / P_Lya
            = 1
            + a_auto exp(lambda_auto k)
            + a_cross cos(k Delta_v) exp(-lambda_cross_abs k)

    The McDonald-like constant-amplitude limit is exactly

        lambda_auto = 0,
        lambda_cross_abs = 0.
    """

    k = np.asarray(k, dtype=float)

    if lambda_auto < 0:
        raise ValueError("lambda_auto must be non-negative.")

    if lambda_cross_abs < 0:
        raise ValueError("lambda_cross_abs must be non-negative.")

    auto_exp = _safe_exp(lambda_auto * k)
    cross_exp = _safe_exp(-lambda_cross_abs * k)

    return (
        1.0
        + a_auto * auto_exp
        + a_cross * np.cos(k * delta_v) * cross_exp
    )


def ma_components(
    k,
    p_lya,
    a_auto,
    a_cross,
    k_auto,
    k_cross,
    delta_v=2271.0,
):
    """
    Return the Ma-model decomposition:

        P_SiIII(k),
        P_Lya-SiIII(k),
        P_tot(k)
    """

    k = np.asarray(k, dtype=float)
    p_lya = np.asarray(p_lya).real

    if np.isfinite(k_auto):
        auto_exp = _safe_exp(k / k_auto)
    else:
        auto_exp = np.ones_like(k, dtype=float)

    if np.isfinite(k_cross):
        cross_exp = _safe_exp(k / k_cross)
    else:
        cross_exp = np.ones_like(k, dtype=float)

    p_auto = a_auto * auto_exp * p_lya

    p_cross = (
        a_cross
        * np.cos(k * delta_v)
        * cross_exp
        * p_lya
    )

    p_tot_model = p_lya + p_auto + p_cross

    return p_auto, p_cross, p_tot_model


def ma_p1d(
    k,
    p_lya,
    a_auto,
    a_cross,
    k_auto,
    k_cross,
    delta_v=2271.0,
):
    """
    Total P1D predicted by the Ma et al. model.
    """

    p_lya = np.asarray(p_lya).real

    return (
        ma_factor(
            k=k,
            a_auto=a_auto,
            a_cross=a_cross,
            k_auto=k_auto,
            k_cross=k_cross,
            delta_v=delta_v,
        )
        * p_lya
    )


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

        model_ratio = (
            1.0
            + a_auto * basis_auto
            + a_cross * basis_cross
        )

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