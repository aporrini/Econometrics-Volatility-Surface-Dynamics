"""
ssvi_surface.py
===============
Parametric SSVI surface calibration for S&P 500 implied volatility.

Model
-----
The 5-parameter surface is:

    theta(T)    = exp(alpha) * T^beta          (ATM total-variance term structure)
    phi(theta)  = eta * theta^(-gamma)         (skew parameter)
    w(k, T)     = (theta/2) * [1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1 - rho^2)]

Parameters: alpha, beta, rho, eta, gamma

Bounds
------
    alpha  in [-10,  2]
    beta   in [  0,  2]
    rho    in [-0.999, 0.999]
    eta    in [1e-4, 50]
    gamma  in [  0,  1]

References
----------
Gatheral, J. & Jacquier, A. (2014) "Arbitrage-free SVI volatility surfaces."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import linregress

from ssvi_calibration import (
    filter_delta_butterfly_continuous,
    SSVI_DELTA_SHORT, SSVI_DELTA_LONG, SSVI_DELTA_DECAY,
    SSVI_MIN_LEFT_POINTS, SSVI_MIN_RIGHT_POINTS, SSVI_MIN_ATM_POINTS,
    SSVI_ATM_BAND, SSVI_FALLBACK_N,
)


# =============================================================================
# 1. Parametric term-structure functions
# =============================================================================

def theta_term_structure(T: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """ATM total-variance term structure: exp(alpha) * T^beta."""
    T = np.asarray(T, dtype=float)
    return np.exp(alpha) * np.power(np.maximum(T, 0.0), beta)


def phi_function(theta: np.ndarray, eta: float, gamma: float) -> np.ndarray:
    """Skew parameter: eta * theta^(-gamma)."""
    theta = np.asarray(theta, dtype=float)
    return eta * np.power(np.maximum(theta, 1e-12), -gamma)


# =============================================================================
# 2. SSVI surface formula
# =============================================================================

def ssvi_total_variance_surface(
    k: np.ndarray,
    T: np.ndarray,
    alpha: float,
    beta: float,
    rho: float,
    eta: float,
    gamma: float,
) -> np.ndarray:
    """SSVI total implied variance w(k, T) for the parametric surface.

    Parameters
    ----------
    k, T  : log-moneyness and time to expiry arrays (same shape)
    alpha, beta, rho, eta, gamma : model parameters

    Returns
    -------
    w  : total implied variance (same shape as k / T)
    """
    k = np.asarray(k, dtype=float)
    T = np.asarray(T, dtype=float)

    theta = theta_term_structure(T, alpha, beta)
    phi   = phi_function(theta, eta, gamma)

    disc = np.maximum((phi * k + rho) ** 2 + (1.0 - rho ** 2), 0.0)
    return (theta / 2.0) * (1.0 + rho * phi * k + np.sqrt(disc))


def ssvi_implied_vol_surface(
    k: np.ndarray,
    T: np.ndarray,
    alpha: float,
    beta: float,
    rho: float,
    eta: float,
    gamma: float,
) -> np.ndarray:
    """Implied volatility from the parametric SSVI surface: sqrt(w / T).

    Returns np.nan where T <= 0 or w <= 0.
    """
    k = np.asarray(k, dtype=float)
    T = np.asarray(T, dtype=float)
    w = ssvi_total_variance_surface(k, T, alpha, beta, rho, eta, gamma)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((w > 0) & (T > 0), np.sqrt(w / T), np.nan)


# =============================================================================
# 3. Data preparation
# =============================================================================

_REQUIRED_COLS = {"TTE"}   # IV and moneyness columns are auto-detected
_USEFUL_COLS = [
    "Time Elapsed",
    "Moneyness", "forward_moneyness",          # use forward_moneyness when present
    "TTE",
    "implied_vol", "implied_vol_forward",      # use forward IV when present
    "Liquidity Factor", "OPEN INTEREST",
]


def prepare_ssvi_surface(
    df: pd.DataFrame,
    time_elapsed,
    min_obs: int         = 100,
    min_iv: float        = 0.03,
    max_iv: float        = 2.00,
    min_tte: float       = 14 / 365,
    max_liquidity: float =  0.30,
    delta_short:  float  = SSVI_DELTA_SHORT,
    delta_long:   float  = SSVI_DELTA_LONG,
    delta_decay:  float  = SSVI_DELTA_DECAY,
    min_left_points:  int   = SSVI_MIN_LEFT_POINTS,
    min_right_points: int   = SSVI_MIN_RIGHT_POINTS,
    min_atm_points:   int   = SSVI_MIN_ATM_POINTS,
    atm_band:         float = SSVI_ATM_BAND,
    n_fallback:       int   = SSVI_FALLBACK_N,
) -> pd.DataFrame:
    """Extract and clean the full IV surface for one date.

    Automatically selects the best available moneyness and IV columns:
    - ``forward_moneyness`` is preferred over ``Moneyness`` when present.
    - ``implied_vol_forward`` is preferred over ``implied_vol`` when present.

    The chosen columns are exposed as ``Moneyness`` and ``implied_vol`` in
    the returned slice so that all downstream functions work unchanged.

    Cleaning steps
    --------------
    1. Select rows: Time Elapsed == time_elapsed
    2. Drop NaN / Inf in critical columns
    3. IV in [min_iv, max_iv]
    4. TTE >= min_tte
    5. Liquidity Factor <= max_liquidity (skipped if column absent)
    6. Side-aware delta butterfly filter with balance enforcement
       (see ``filter_delta_butterfly_continuous``)

    The slice carries the delta-filter balance diagnostics in ``.attrs``:
    ``n_left_put_otm``, ``n_right_call_otm``, ``n_atm``, ``flag_unbalanced_smile``.

    Raises
    ------
    KeyError   required columns missing
    ValueError fewer than min_obs rows survive
    """
    # ── Auto-detect best available columns ───────────────────────────────────
    k_col  = ("forward_moneyness"   if "forward_moneyness"   in df.columns else "Moneyness")
    iv_col = ("implied_vol_forward" if "implied_vol_forward" in df.columns else "implied_vol")

    missing = (_REQUIRED_COLS | {k_col, iv_col}) - set(df.columns)
    if missing:
        raise KeyError(f"Required columns missing: {missing}")

    present = list(dict.fromkeys(
        [c for c in _USEFUL_COLS if c in df.columns] + [k_col, iv_col]
    ))
    sl = df.loc[df["Time Elapsed"] == time_elapsed, present].copy()

    sl = sl.replace([np.inf, -np.inf], np.nan).dropna(subset=[k_col, "TTE", iv_col])
    sl = sl[(sl[iv_col] >= min_iv) & (sl[iv_col] <= max_iv)]
    sl = sl[sl["TTE"] >= min_tte]

    if "Liquidity Factor" in sl.columns:
        sl = sl[sl["Liquidity Factor"].fillna(np.inf) <= max_liquidity]

    sl = sl.reset_index(drop=True)

    balance_attrs = dict(n_left_put_otm=0, n_right_call_otm=0, n_atm=0,
                         flag_unbalanced_smile=True)
    if len(sl) > 0:
        sl = filter_delta_butterfly_continuous(
            sl,
            delta_short=delta_short, delta_long=delta_long, decay=delta_decay,
            min_left_points=min_left_points, min_right_points=min_right_points,
            min_atm_points=min_atm_points, atm_band=atm_band, n_fallback=n_fallback,
        )
        balance_attrs = {k: sl.attrs.get(k, v) for k, v in balance_attrs.items()}

    sl = sl.reset_index(drop=True)

    if len(sl) < min_obs:
        raise ValueError(
            f"Only {len(sl)} obs for {time_elapsed!r} after cleaning — need >= {min_obs}."
        )

    # ── Standardise names so downstream code always uses "Moneyness" / "implied_vol"
    if k_col != "Moneyness":
        sl["Moneyness"] = sl[k_col]
    if iv_col != "implied_vol":
        sl["implied_vol"] = sl[iv_col]

    sl.attrs.update(balance_attrs)
    return sl


# =============================================================================
# 4. Weights
# =============================================================================

def build_weights(surface_df: pd.DataFrame) -> np.ndarray:
    """Normalised weights = log1p(OI) / (1 + LF), mean-normalised."""
    n  = len(surface_df)
    lf = (
        surface_df["Liquidity Factor"].fillna(0.0).clip(lower=0.0).to_numpy(float)
        if "Liquidity Factor" in surface_df.columns
        else np.zeros(n)
    )
    oi = (
        surface_df["OPEN INTEREST"].fillna(0.0).clip(lower=0.0).to_numpy(float)
        if "OPEN INTEREST" in surface_df.columns
        else np.ones(n)
    )
    w     = np.log1p(oi) / (1.0 + lf)
    mean_w = w.mean()
    return w / (mean_w + 1e-8) if mean_w > 1e-8 else np.ones(n)


# =============================================================================
# 5. Finance-consistent initial guess
# =============================================================================

def initial_guess_ssvi(surface_df: pd.DataFrame) -> np.ndarray:
    """Data-driven starting point for (alpha, beta, rho, eta, gamma).

    alpha, beta  Fit log(w_ATM) = alpha + beta * log(T) via OLS on ATM options.
    rho          Fixed at -0.5 (typical SPX skew).
    eta          Set so that phi(theta_median) ≈ 1.
    gamma        Fixed at 0.5 (midpoint of [0, 1]).
    """
    sl    = surface_df
    T     = sl["TTE"].to_numpy(float)
    w_obs = sl["implied_vol"].to_numpy(float) ** 2 * T

    # alpha, beta from log-log regression of ATM total variance vs T
    atm = np.abs(sl["Moneyness"].to_numpy(float)) < 0.05
    if atm.sum() >= 5:
        log_T = np.log(np.maximum(T[atm], 1e-8))
        log_w = np.log(np.maximum(w_obs[atm], 1e-8))
        slope, intercept, *_ = linregress(log_T, log_w)
        beta0  = float(np.clip(slope, 0.05, 1.95))
        alpha0 = float(np.clip(intercept, -9.0, 1.9))
    else:
        beta0  = 1.0
        alpha0 = np.log(max(float(np.median(w_obs)), 1e-6))

    rho0   = -0.5
    gamma0 = 0.5

    # eta: phi(theta) = eta * theta^(-gamma) ≈ 1 at median theta
    theta_med = float(np.exp(alpha0) * np.median(T) ** beta0)
    theta_med = max(theta_med, 1e-6)
    eta0      = float(np.clip(theta_med ** gamma0, 1e-3, 49.0))

    return np.array([alpha0, beta0, rho0, eta0, gamma0])


# =============================================================================
# 6. Objective function
# =============================================================================

# Parameter bounds: (alpha, beta, rho, eta, gamma)
_LO = np.array([-10.0,  0.0, -0.999, 1e-4, 0.0])
_HI = np.array([  2.0,  2.0,  0.999, 50.0, 1.0])

_PENALTY = 100.0


def ssvi_residuals(
    params: np.ndarray,
    k: np.ndarray,
    T: np.ndarray,
    w_obs: np.ndarray,
    weights: np.ndarray | None = None,
    use_arbitrage_penalty: bool = False,
) -> np.ndarray:
    """Weighted residuals in total-variance space for scipy least_squares.

    Residual vector
    ---------------
    [ sqrt(w_i) * (w_model_i - w_obs_i), ..., pen_rho ]

    Soft penalty for |rho| near 1 appended as extra scalar residual.
    """
    alpha, beta, rho, eta, gamma = params

    if abs(rho) >= 0.999 or eta <= 0:
        return np.full(len(k) + 1, 1e4)

    w_model   = ssvi_total_variance_surface(k, T, alpha, beta, rho, eta, gamma)
    residuals = np.where(np.isfinite(w_model), w_model - w_obs, 1e3)

    if weights is not None:
        residuals = residuals * np.sqrt(np.maximum(weights, 0.0))

    pen_rho = _PENALTY * max(abs(rho) - 0.95, 0.0)

    if use_arbitrage_penalty:
        theta_g = theta_term_structure(T, alpha, beta)
        phi_g   = phi_function(theta_g, eta, gamma)
        abs_rho = abs(rho)
        cond1   = theta_g * phi_g * (1.0 + abs_rho)
        cond2   = theta_g * phi_g ** 2 * (1.0 + abs_rho)
        pen_arb = 10.0 * float(np.sqrt(np.mean(
            np.maximum(cond1 - 4.0, 0.0) ** 2 + np.maximum(cond2 - 4.0, 0.0) ** 2
        )))
    else:
        pen_arb = 0.0

    return np.append(residuals, [pen_rho, pen_arb])


# =============================================================================
# 7. Calibration
# =============================================================================

def calibrate_ssvi_surface(
    df: pd.DataFrame,
    time_elapsed,
    use_weights: bool    = True,
    min_obs: int         = 100,
    min_iv: float        = 0.03,
    max_iv: float        = 2.00,
    min_tte: float       = 14 / 365,
    max_liquidity: float =  0.30,
    delta_short:  float  = SSVI_DELTA_SHORT,
    delta_long:   float  = SSVI_DELTA_LONG,
    delta_decay:  float  = SSVI_DELTA_DECAY,
    min_left_points:  int   = SSVI_MIN_LEFT_POINTS,
    min_right_points: int   = SSVI_MIN_RIGHT_POINTS,
    min_atm_points:   int   = SSVI_MIN_ATM_POINTS,
    atm_band:         float = SSVI_ATM_BAND,
    n_fallback:       int   = SSVI_FALLBACK_N,
    use_arbitrage_penalty: bool = False,
) -> dict:
    """Calibrate the 5-parameter SSVI surface for one date.

    Returns
    -------
    dict with keys:
        time_elapsed, alpha, beta, rho, eta, gamma,
        success, cost, rmse_w, rmse_iv, mae_iv, n_obs,
        tte_range, moneyness_range,
        n_left_put_otm, n_right_call_otm, n_atm, flag_unbalanced_smile,
        message
    """
    base = dict(
        time_elapsed=time_elapsed,
        alpha=np.nan, beta=np.nan, rho=np.nan, eta=np.nan, gamma=np.nan,
        success=False, cost=np.nan,
        rmse_w=np.nan, rmse_iv=np.nan, mae_iv=np.nan,
        n_obs=0, tte_range=(np.nan, np.nan), moneyness_range=(np.nan, np.nan),
        n_left_put_otm=0, n_right_call_otm=0, n_atm=0, flag_unbalanced_smile=True,
        message="",
    )

    # ── 1. Prepare surface slice ──────────────────────────────────────────────
    try:
        sl = prepare_ssvi_surface(
            df, time_elapsed,
            min_obs=min_obs, min_iv=min_iv, max_iv=max_iv,
            min_tte=min_tte, max_liquidity=max_liquidity,
            delta_short=delta_short, delta_long=delta_long, delta_decay=delta_decay,
            min_left_points=min_left_points, min_right_points=min_right_points,
            min_atm_points=min_atm_points, atm_band=atm_band, n_fallback=n_fallback,
        )
    except (ValueError, KeyError) as exc:
        base["message"] = str(exc)
        return base

    k     = sl["Moneyness"].to_numpy(float)
    T     = sl["TTE"].to_numpy(float)
    w_obs = sl["implied_vol"].to_numpy(float) ** 2 * T

    base["n_obs"]           = len(sl)
    base["tte_range"]       = (float(T.min()), float(T.max()))
    base["moneyness_range"] = (float(k.min()), float(k.max()))
    base["n_left_put_otm"]        = sl.attrs.get("n_left_put_otm", 0)
    base["n_right_call_otm"]      = sl.attrs.get("n_right_call_otm", 0)
    base["n_atm"]                 = sl.attrs.get("n_atm", 0)
    base["flag_unbalanced_smile"] = sl.attrs.get("flag_unbalanced_smile", True)

    # ── 2. Weights ────────────────────────────────────────────────────────────
    weights = build_weights(sl) if use_weights else None

    # ── 3. Optimise ───────────────────────────────────────────────────────────
    x0 = initial_guess_ssvi(sl)

    try:
        opt = least_squares(
            ssvi_residuals,
            x0,
            args=(k, T, w_obs, weights, use_arbitrage_penalty),
            bounds=(_LO, _HI),
            method="trf",
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
            max_nfev=20_000,
        )
    except Exception as exc:
        base["message"] = f"Optimiser error: {exc}"
        return base

    alpha_h, beta_h, rho_h, eta_h, gamma_h = opt.x

    # ── 4. Diagnostics ────────────────────────────────────────────────────────
    w_fit   = ssvi_total_variance_surface(k, T, alpha_h, beta_h, rho_h, eta_h, gamma_h)
    sig_fit = ssvi_implied_vol_surface(k, T, alpha_h, beta_h, rho_h, eta_h, gamma_h)
    sig_obs = sl["implied_vol"].to_numpy(float)

    ok_w  = np.isfinite(w_fit)
    ok_iv = np.isfinite(sig_fit)
    rmse_w  = float(np.sqrt(np.mean((w_fit[ok_w]   - w_obs[ok_w])   ** 2))) if ok_w.any()  else np.nan
    rmse_iv = float(np.sqrt(np.mean((sig_fit[ok_iv] - sig_obs[ok_iv]) ** 2))) if ok_iv.any() else np.nan
    mae_iv  = float(np.mean(np.abs(sig_fit[ok_iv] - sig_obs[ok_iv])))         if ok_iv.any() else np.nan

    return dict(
        time_elapsed=time_elapsed,
        alpha=float(alpha_h), beta=float(beta_h), rho=float(rho_h),
        eta=float(eta_h), gamma=float(gamma_h),
        success=bool(opt.success), cost=float(opt.cost),
        rmse_w=rmse_w, rmse_iv=rmse_iv, mae_iv=mae_iv,
        n_obs=len(sl),
        tte_range=(float(T.min()), float(T.max())),
        moneyness_range=(float(k.min()), float(k.max())),
        n_left_put_otm=base["n_left_put_otm"],
        n_right_call_otm=base["n_right_call_otm"],
        n_atm=base["n_atm"],
        flag_unbalanced_smile=base["flag_unbalanced_smile"],
        message=opt.message,
    )


# =============================================================================
# 8. Predictions
# =============================================================================

def predict_ssvi_surface(
    surface_df: pd.DataFrame,
    result: dict,
) -> pd.DataFrame:
    """Attach SSVI surface predictions to a slice.

    Added columns
    -------------
    w_obs      : implied_vol^2 * TTE
    ssvi_w     : w(k, T) from the parametric model
    ssvi_iv    : sqrt(ssvi_w / T)
    ssvi_err_iv: ssvi_iv - implied_vol
    ssvi_err_w : ssvi_w  - w_obs
    """
    df = surface_df.copy()
    alpha = result["alpha"]
    beta  = result["beta"]
    rho   = result["rho"]
    eta   = result["eta"]
    gamma = result["gamma"]

    k = df["Moneyness"].to_numpy(float)
    T = df["TTE"].to_numpy(float)

    df["w_obs"]      = df["implied_vol"].to_numpy(float) ** 2 * T
    df["ssvi_w"]     = ssvi_total_variance_surface(k, T, alpha, beta, rho, eta, gamma)
    df["ssvi_iv"]    = ssvi_implied_vol_surface(k, T, alpha, beta, rho, eta, gamma)
    df["ssvi_err_iv"] = df["ssvi_iv"] - df["implied_vol"]
    df["ssvi_err_w"]  = df["ssvi_w"]  - df["w_obs"]
    return df


# =============================================================================
# 9. Diagnostics report
# =============================================================================

def print_ssvi_diagnostics(result: dict) -> None:
    """Print a structured parameter and fit-quality report."""
    sep = "─" * 60
    print(sep)
    print(f"  SSVI Surface — {result.get('time_elapsed', '?')}")
    print(sep)
    print(f"  n_obs   = {result.get('n_obs', 0):,}")
    tte = result.get("tte_range", (np.nan, np.nan))
    mn  = result.get("moneyness_range", (np.nan, np.nan))
    print(f"  TTE     : [{tte[0]*365:.0f}d, {tte[1]*365:.0f}d]")
    print(f"  k range : [{mn[0]:.3f}, {mn[1]:.3f}]")
    print()

    alpha = result.get("alpha", np.nan)
    beta  = result.get("beta",  np.nan)
    rho   = result.get("rho",   np.nan)
    eta   = result.get("eta",   np.nan)
    gamma = result.get("gamma", np.nan)

    print(f"  alpha   = {alpha:.4f}   (level: exp(alpha) = {np.exp(alpha):.4f})")
    print(f"  beta    = {beta:.4f}   (slope: theta(T) ~ T^beta)")
    print(f"  rho     = {rho:.4f}   ({'OK' if rho < 0 else 'WARNING: expected rho < 0 for SPX'})")
    print(f"  eta     = {eta:.4f}   (skew scale)")
    print(f"  gamma   = {gamma:.4f}   (skew decay in theta)")
    print()

    rmse_w  = result.get("rmse_w",  np.nan)
    rmse_iv = result.get("rmse_iv", np.nan)
    mae_iv  = result.get("mae_iv",  np.nan)
    ok_rmse = "OK" if np.isfinite(rmse_iv) and rmse_iv < 0.01 else "LARGE"
    print(f"  RMSE_w  = {rmse_w:.6f}")
    print(f"  RMSE_iv = {rmse_iv:.6f}  [{ok_rmse}]")
    print(f"  MAE_iv  = {mae_iv:.6f}")
    print()

    if result.get("success"):
        print("  Converged: YES")
    else:
        print(f"  Converged: NO — {result.get('message', '')}")

    # Warnings
    warnings = []
    if np.isfinite(rho) and rho >= 0:
        warnings.append("rho >= 0 — SPX skew is typically negative")
    if np.isfinite(rmse_iv) and rmse_iv > 0.02:
        warnings.append(f"RMSE_iv = {rmse_iv:.4f} > 0.02 — poor fit")
    if np.isfinite(beta) and beta > 1.5:
        warnings.append(f"beta = {beta:.3f} > 1.5 — very steep term structure")

    print(sep)
    if warnings:
        for w in warnings:
            print(f"  ⚠  {w}")
    else:
        print("  ✓  All checks passed")
    print()


# =============================================================================
# 10. Visualisation
# =============================================================================

def plot_ssvi_surface_fit(
    pred_df: pd.DataFrame,
    result: dict,
    title: str | None = None,
) -> None:
    """Scatter of observed IV and fitted SSVI curves coloured by TTE.

    Two panels: left = IV vs moneyness, right = total variance vs moneyness.
    Each panel overlays fitted SSVI curves at 10th/50th/90th TTE percentiles.
    """
    df = pred_df.copy()
    if "ssvi_iv" not in df.columns:
        df = predict_ssvi_surface(df, result)

    alpha = result["alpha"]
    beta  = result["beta"]
    rho   = result["rho"]
    eta   = result["eta"]
    gamma = result["gamma"]

    tte_vals  = df["TTE"].to_numpy(float)
    tte_min   = tte_vals.min()
    tte_range = max(tte_vals.max() - tte_min, 1e-8)
    tte_norm  = (tte_vals - tte_min) / tte_range
    cmap      = plt.cm.plasma

    k_grid   = np.linspace(df["Moneyness"].min(), df["Moneyness"].max(), 300)
    tte_pcts = np.percentile(tte_vals, [10, 50, 90])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    suptitle = title or (
        f"Parametric SSVI Surface — {result.get('time_elapsed', '')}\n"
        f"α={alpha:.3f}  β={beta:.3f}  ρ={rho:.3f}  "
        f"η={eta:.3f}  γ={gamma:.3f}   "
        f"RMSE_iv={result.get('rmse_iv', np.nan):.4f}  "
        f"n={result.get('n_obs', 0):,}"
    )
    fig.suptitle(suptitle, fontsize=10)

    # ── IV panel ──────────────────────────────────────────────────────────────
    sc = ax1.scatter(df["Moneyness"], df["implied_vol"],
                     c=tte_norm, cmap=cmap, s=8, alpha=0.4, zorder=2,
                     label="Observed IV")
    plt.colorbar(sc, ax=ax1, label="TTE (normalised)")
    for tte_v in tte_pcts:
        col     = cmap((tte_v - tte_min) / tte_range)
        iv_line = ssvi_implied_vol_surface(
            k_grid, np.full_like(k_grid, tte_v), alpha, beta, rho, eta, gamma
        )
        ax1.plot(k_grid, iv_line, color=col, linewidth=2.0,
                 label=f"T={tte_v * 365:.0f}d")
    ax1.set_xlabel("log(K/S)")
    ax1.set_ylabel("Implied volatility")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(alpha=0.3)

    # ── Total-variance panel ──────────────────────────────────────────────────
    ax2.scatter(df["Moneyness"], df["w_obs"],
                c=tte_norm, cmap=cmap, s=8, alpha=0.4, zorder=2)
    for tte_v in tte_pcts:
        col    = cmap((tte_v - tte_min) / tte_range)
        w_line = ssvi_total_variance_surface(
            k_grid, np.full_like(k_grid, tte_v), alpha, beta, rho, eta, gamma
        )
        ax2.plot(k_grid, w_line, color=col, linewidth=2.0,
                 label=f"T={tte_v * 365:.0f}d")
    ax2.set_xlabel("log(K/S)")
    ax2.set_ylabel("Total variance w = σ²T")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_ssvi_errors(pred_df: pd.DataFrame) -> None:
    """Three-panel error diagnostic: histogram, errors vs moneyness, vs TTE."""
    df = pred_df.copy()
    err = df["ssvi_err_iv"].dropna()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("SSVI IV error diagnostics  (ssvi_iv − observed_iv)", fontsize=10)

    # Histogram
    ax1.hist(err, bins=50, edgecolor="white", linewidth=0.3, color="steelblue")
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("Error (IV units)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Distribution   MAE={err.abs().mean():.4f}")
    ax1.grid(alpha=0.3)

    # vs Moneyness
    ax2.scatter(df["Moneyness"], df["ssvi_err_iv"], s=5, alpha=0.4, color="steelblue")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("log(K/S)")
    ax2.set_ylabel("Error")
    ax2.set_title("Error vs Moneyness")
    ax2.grid(alpha=0.3)

    # vs TTE
    ax3.scatter(df["TTE"] * 365, df["ssvi_err_iv"], s=5, alpha=0.4, color="steelblue")
    ax3.axhline(0, color="black", linewidth=0.8)
    ax3.set_xlabel("TTE (days)")
    ax3.set_ylabel("Error")
    ax3.set_title("Error vs TTE")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 11. Multi-date calibration
# =============================================================================

def calibrate_all_dates(
    df: pd.DataFrame,
    min_obs: int         = 100,
    use_weights: bool    = True,
    verbose: bool        = True,
    min_iv: float        = 0.03,
    max_iv: float        = 2.00,
    min_tte: float       = 14 / 365,
    max_liquidity: float =  0.30,
    delta_short:  float  = SSVI_DELTA_SHORT,
    delta_long:   float  = SSVI_DELTA_LONG,
    delta_decay:  float  = SSVI_DELTA_DECAY,
    min_left_points:  int   = SSVI_MIN_LEFT_POINTS,
    min_right_points: int   = SSVI_MIN_RIGHT_POINTS,
    min_atm_points:   int   = SSVI_MIN_ATM_POINTS,
    atm_band:         float = SSVI_ATM_BAND,
    n_fallback:       int   = SSVI_FALLBACK_N,
    use_arbitrage_penalty: bool = False,
) -> pd.DataFrame:
    """Calibrate the parametric SSVI surface for every date in the dataset.

    Parameters
    ----------
    df          : DataFrame with columns including 'Time Elapsed' and 'implied_vol'
    min_obs     : minimum observations required per date (skipped if fewer)
    use_weights : use OI / liquidity weights in the objective
    verbose     : print progress every 10 dates

    Returns
    -------
    DataFrame with one row per date, columns:
        time_elapsed, alpha, beta, rho, eta, gamma,
        success, cost, rmse_w, rmse_iv, mae_iv,
        n_obs, tte_range, moneyness_range,
        n_left_put_otm, n_right_call_otm, n_atm, flag_unbalanced_smile,
        message
    """
    dates = sorted(df["Time Elapsed"].unique())
    n     = len(dates)
    rows  = []

    for i, te in enumerate(dates, 1):
        result = calibrate_ssvi_surface(
            df, te,
            use_weights           = use_weights,
            min_obs               = min_obs,
            min_iv                = min_iv,
            max_iv                = max_iv,
            min_tte               = min_tte,
            max_liquidity         = max_liquidity,
            delta_short           = delta_short,
            delta_long            = delta_long,
            delta_decay           = delta_decay,
            min_left_points       = min_left_points,
            min_right_points      = min_right_points,
            min_atm_points        = min_atm_points,
            atm_band              = atm_band,
            n_fallback            = n_fallback,
            use_arbitrage_penalty = use_arbitrage_penalty,
        )

        # Flatten tuple columns for CSV compatibility
        result["tte_min"]       = result["tte_range"][0]
        result["tte_max"]       = result["tte_range"][1]
        result["moneyness_min"] = result["moneyness_range"][0]
        result["moneyness_max"] = result["moneyness_range"][1]
        del result["tte_range"], result["moneyness_range"]

        rows.append(result)

        if verbose and (i % 10 == 0 or i == n):
            n_ok = sum(r["success"] for r in rows)
            print(f"Done {i:>4}/{n} dates   "
                  f"success {n_ok}/{i} ({100*n_ok/i:.0f}%)")

    results_df = pd.DataFrame(rows)
    if verbose:
        n_ok = results_df["success"].sum()
        print(f"\nFinished: {n_ok}/{n} dates calibrated successfully "
              f"({100*n_ok/n:.1f}%)")
    return results_df


# =============================================================================
# 12. Interactive 3D surface (Plotly)
# =============================================================================

def plot_interactive_ssvi_surface(
    pred_df: pd.DataFrame,
    result: dict,
    n_k: int = 60,
    n_t: int = 60,
) -> None:
    """Interactive Plotly 3D surface: fitted SSVI + observed IV scatter.

    Parameters
    ----------
    pred_df : DataFrame with columns Moneyness, TTE, implied_vol
              (output of predict_ssvi_surface)
    result  : calibration result dict (alpha, beta, rho, eta, gamma)
    n_k     : grid points along moneyness axis
    n_t     : grid points along TTE axis
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required: pip install plotly")

    alpha = result["alpha"]
    beta  = result["beta"]
    rho   = result["rho"]
    eta   = result["eta"]
    gamma = result["gamma"]

    # ── Grid ─────────────────────────────────────────────────────────────────
    k_min = float(pred_df["Moneyness"].min())
    k_max = float(pred_df["Moneyness"].max())
    t_min = float(pred_df["TTE"].min())
    t_max = float(pred_df["TTE"].max())

    k_grid = np.linspace(k_min, k_max, n_k)
    t_grid = np.linspace(t_min, t_max, n_t)
    KK, TT = np.meshgrid(k_grid, t_grid)

    ZZ = ssvi_implied_vol_surface(KK, TT, alpha, beta, rho, eta, gamma)

    # ── Traces ────────────────────────────────────────────────────────────────
    surface_trace = go.Surface(
        x=KK,
        y=TT * 365,   # display in days
        z=ZZ,
        colorscale="Plasma",
        opacity=0.72,
        showscale=True,
        colorbar=dict(title="IV", thickness=14),
        name="SSVI fit",
        hovertemplate="k=%{x:.3f}<br>TTE=%{y:.0f}d<br>IV=%{z:.4f}<extra>SSVI fit</extra>",
    )

    scatter_trace = go.Scatter3d(
        x=pred_df["Moneyness"].to_numpy(),
        y=pred_df["TTE"].to_numpy() * 365,
        z=pred_df["implied_vol"].to_numpy(),
        mode="markers",
        marker=dict(
            size=2.0,
            color="black",
            opacity=0.55,
        ),
        name="Observed IV",
        hovertemplate="k=%{x:.3f}<br>TTE=%{y:.0f}d<br>IV=%{z:.4f}<extra>Observed</extra>",
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    title_text = (
        f"SSVI Surface — date {result.get('time_elapsed', '')}   "
        f"ρ={rho:.3f}  η={eta:.3f}  γ={gamma:.3f}  "
        f"RMSE_iv={result.get('rmse_iv', float('nan')):.4f}  "
        f"n={result.get('n_obs', 0):,}"
    )

    fig = go.Figure(data=[surface_trace, scatter_trace])
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=13)),
        scene=dict(
            xaxis=dict(title="log(K/S)", showgrid=True, gridcolor="lightgrey"),
            yaxis=dict(title="TTE (days)", showgrid=True, gridcolor="lightgrey"),
            zaxis=dict(title="Implied volatility", showgrid=True, gridcolor="lightgrey"),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
        ),
        legend=dict(x=0.01, y=0.99),
        width=950,
        height=720,
        margin=dict(l=0, r=0, t=60, b=0),
    )
    fig.show()


# =============================================================================
# 13. Animated surface evolution (Plotly)
# =============================================================================

def plot_ssvi_surface_animation(
    df_iv: pd.DataFrame,
    results_df: pd.DataFrame,
    dates=None,
    n_k: int = 40,
    n_t: int = 40,
    step: int = 5,
) -> None:
    """Plotly animation of the SSVI surface evolving across dates.

    Parameters
    ----------
    df_iv       : full IV dataset (used for global grid bounds)
    results_df  : output of calibrate_all_dates()
    dates       : explicit list of Time Elapsed values to animate;
                  if None, every `step`-th successful date is used
    n_k, n_t    : grid resolution (lower = faster rendering)
    step        : subsample step when dates is None (default every 5th date)
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required: pip install plotly")

    ok = results_df[results_df["success"]].copy()

    if dates is None:
        all_ok = sorted(ok["time_elapsed"].unique())
        dates  = all_ok[::step]

    ok = ok[ok["time_elapsed"].isin(dates)].set_index("time_elapsed")

    # ── Global grid (fixed across frames) ────────────────────────────────────
    k_lo = float(df_iv["Moneyness"].quantile(0.02))
    k_hi = float(df_iv["Moneyness"].quantile(0.98))
    t_lo = float(df_iv["TTE"].quantile(0.02))
    t_hi = float(df_iv["TTE"].quantile(0.98))

    k_grid = np.linspace(k_lo, k_hi, n_k)
    t_grid = np.linspace(t_lo, t_hi, n_t)
    KK, TT = np.meshgrid(k_grid, t_grid)
    TT_days = TT * 365

    def _surface_for_date(te):
        row = ok.loc[te]
        ZZ  = ssvi_implied_vol_surface(
            KK, TT,
            row["alpha"], row["beta"], row["rho"],
            row["eta"],   row["gamma"],
        )
        return ZZ

    # Global z-range for a fixed colour axis
    z_vals = []
    for te in dates:
        if te in ok.index:
            z_vals.append(_surface_for_date(te))
    z_all   = np.concatenate([z[np.isfinite(z)] for z in z_vals])
    zmin    = float(np.percentile(z_all, 1))
    zmax    = float(np.percentile(z_all, 99))

    # ── Build frames ─────────────────────────────────────────────────────────
    frames = []
    for te in dates:
        if te not in ok.index:
            continue
        row = ok.loc[te]
        ZZ  = _surface_for_date(te)
        frames.append(go.Frame(
            data=[go.Surface(
                x=KK, y=TT_days, z=ZZ,
                colorscale="Plasma",
                cmin=zmin, cmax=zmax,
                opacity=0.85,
                showscale=False,
            )],
            name=str(te),
            layout=go.Layout(title_text=(
                f"SSVI Surface — date {te}  |  "
                f"ρ={row['rho']:.3f}  η={row['eta']:.3f}  "
                f"γ={row['gamma']:.3f}  RMSE_iv={row['rmse_iv']:.4f}"
            )),
        ))

    if not frames:
        print("No valid dates found for animation.")
        return

    # ── Initial frame ─────────────────────────────────────────────────────────
    ZZ0  = _surface_for_date(dates[0])
    row0 = ok.loc[dates[0]]

    fig = go.Figure(
        data=[go.Surface(
            x=KK, y=TT_days, z=ZZ0,
            colorscale="Plasma",
            cmin=zmin, cmax=zmax,
            opacity=0.85,
            showscale=True,
            colorbar=dict(title="IV", thickness=14),
        )],
        frames=frames,
    )

    slider_steps = [
        dict(
            args=[[str(d)],
                  {"frame": {"duration": 300, "redraw": True}, "mode": "immediate"}],
            label=str(d),
            method="animate",
        )
        for d in dates if d in ok.index
    ]

    fig.update_layout(
        title=dict(
            text=(
                f"SSVI Surface Evolution — {len(frames)} dates  "
                f"(every {step}th successful day)"
            ),
            font=dict(size=13),
        ),
        scene=dict(
            xaxis=dict(title="log(K/S)"),
            yaxis=dict(title="TTE (days)"),
            zaxis=dict(title="Implied vol", range=[zmin, zmax]),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.05, x=0.0, xanchor="left",
            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[None, {"frame": {"duration": 300, "redraw": True},
                                 "fromcurrent": True, "transition": {"duration": 150}}],
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                ),
            ],
        )],
        sliders=[dict(
            active=0,
            steps=slider_steps,
            transition={"duration": 150},
            x=0, len=1.0,
            currentvalue=dict(prefix="Date: ", font=dict(size=12)),
        )],
        width=980,
        height=740,
        margin=dict(l=0, r=0, t=80, b=60),
    )
    fig.show()


# =============================================================================
# 14. No-arbitrage checks (Gatheral & Jacquier 2014)
# =============================================================================

def check_butterfly_arbitrage(
    result: dict,
    T_grid: np.ndarray | None = None,
    n_T: int = 200,
) -> dict:
    """Check sufficient conditions for the absence of butterfly arbitrage.

    For the parametric SSVI surface, butterfly arbitrage is absent when:
        cond1(T) = theta(T) * phi(T) * (1 + |rho|) < 4
        cond2(T) = theta(T) * phi(T)^2 * (1 + |rho|) < 4

    for all T in the calibration range.

    Returns
    -------
    dict with keys: butterfly_ok, max_cond1, max_cond2, n_violations,
                    pct_violations, T_grid, cond1, cond2
    """
    alpha = result["alpha"]
    beta  = result["beta"]
    rho   = result["rho"]
    eta   = result["eta"]
    gamma = result["gamma"]

    tte_min, tte_max = result.get("tte_range", (1 / 365, 3.0))

    if T_grid is None:
        T_grid = np.linspace(max(tte_min, 1 / 365), tte_max, n_T)
    T_grid = np.asarray(T_grid, dtype=float)

    theta   = theta_term_structure(T_grid, alpha, beta)
    phi     = phi_function(theta, eta, gamma)
    abs_rho = abs(rho)

    cond1 = theta * phi * (1.0 + abs_rho)
    cond2 = theta * phi ** 2 * (1.0 + abs_rho)

    violated = (cond1 >= 4.0) | (cond2 >= 4.0)
    n_viol   = int(violated.sum())

    return dict(
        butterfly_ok   = n_viol == 0,
        max_cond1      = float(np.max(cond1)),
        max_cond2      = float(np.max(cond2)),
        n_violations   = n_viol,
        pct_violations = 100.0 * n_viol / len(T_grid),
        T_grid         = T_grid,
        cond1          = cond1,
        cond2          = cond2,
    )


def check_calendar_arbitrage(
    result: dict,
    k_grid: np.ndarray | None = None,
    T_grid: np.ndarray | None = None,
    n_k: int = 100,
    n_T: int = 100,
    tolerance: float = 1e-8,
) -> dict:
    """Check for calendar arbitrage in the SSVI surface.

    Calendar arbitrage is absent when total implied variance w(k, T) is
    non-decreasing in T for every log-moneyness k:

        w(k, T_{i+1}) - w(k, T_i) >= -tolerance   for all k, i

    Returns
    -------
    dict with keys: calendar_ok, n_violations, pct_violations,
                    worst_negative_diff, k_grid, T_grid, dw
    """
    alpha = result["alpha"]
    beta  = result["beta"]
    rho   = result["rho"]
    eta   = result["eta"]
    gamma = result["gamma"]

    tte_min, tte_max = result.get("tte_range", (1 / 365, 3.0))

    if k_grid is None:
        k_grid = np.linspace(-0.3, 0.3, n_k)
    if T_grid is None:
        T_grid = np.linspace(max(tte_min, 1 / 365), tte_max, n_T)

    k_grid = np.asarray(k_grid, dtype=float)
    T_grid = np.asarray(T_grid, dtype=float)

    KK, TT = np.meshgrid(k_grid, T_grid, indexing="ij")
    W  = ssvi_total_variance_surface(KK, TT, alpha, beta, rho, eta, gamma)
    dW = np.diff(W, axis=1)   # shape (n_k, n_T - 1)

    violated = dW < -tolerance
    n_viol   = int(violated.sum())
    worst    = float(dW[violated].min()) if n_viol > 0 else 0.0

    return dict(
        calendar_ok         = n_viol == 0,
        n_violations        = n_viol,
        pct_violations      = 100.0 * n_viol / dW.size if dW.size > 0 else 0.0,
        worst_negative_diff = worst,
        k_grid              = k_grid,
        T_grid              = T_grid,
        dw                  = dW,
    )


def run_no_arbitrage_checks(
    result: dict,
    n_T_butterfly: int = 200,
    n_k: int = 100,
    n_T_calendar: int = 100,
    tolerance: float = 1e-8,
) -> dict:
    """Run butterfly and calendar arbitrage checks and combine results.

    Returns
    -------
    dict combining both check outputs, plus:
        no_arbitrage : bool — True only if both checks pass
        _bf, _cal    : raw dicts from the individual checks (for plots)
    """
    bf  = check_butterfly_arbitrage(result, n_T=n_T_butterfly)
    cal = check_calendar_arbitrage(result, n_k=n_k, n_T=n_T_calendar,
                                   tolerance=tolerance)

    return dict(
        butterfly_ok              = bf["butterfly_ok"],
        max_cond1                 = bf["max_cond1"],
        max_cond2                 = bf["max_cond2"],
        n_butterfly_violations    = bf["n_violations"],
        pct_butterfly_violations  = bf["pct_violations"],
        calendar_ok               = cal["calendar_ok"],
        n_calendar_violations     = cal["n_violations"],
        pct_calendar_violations   = cal["pct_violations"],
        worst_negative_diff       = cal["worst_negative_diff"],
        no_arbitrage              = bf["butterfly_ok"] and cal["calendar_ok"],
        _bf                       = bf,
        _cal                      = cal,
    )


def run_no_arbitrage_checks_all(
    results_df: pd.DataFrame,
    n_k: int = 100,
    n_T: int = 200,
    tolerance: float = 1e-8,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run no-arbitrage checks for every successfully calibrated date.

    Parameters
    ----------
    results_df  : output of calibrate_all_dates()
    n_k, n_T    : grid resolution (n_T used for both butterfly and calendar)
    tolerance   : numerical tolerance for calendar check
    verbose     : print progress every 50 dates and final summary

    Returns
    -------
    DataFrame with one row per successful date and columns:
        time_elapsed, butterfly_ok, calendar_ok, no_arbitrage,
        max_cond1, max_cond2, pct_butterfly_violations,
        n_calendar_violations, pct_calendar_violations, worst_negative_diff
    """
    ok = results_df[results_df["success"]].reset_index(drop=True)
    records = []

    for i, row in ok.iterrows():
        result = row.to_dict()
        # calibrate_all_dates flattens tte_range → tte_min / tte_max
        if "tte_range" not in result:
            result["tte_range"] = (
                float(result.get("tte_min", 1 / 365)),
                float(result.get("tte_max", 3.0)),
            )

        checks = run_no_arbitrage_checks(
            result,
            n_T_butterfly=n_T,
            n_k=n_k,
            n_T_calendar=n_T,
            tolerance=tolerance,
        )

        records.append(dict(
            time_elapsed             = row["time_elapsed"],
            butterfly_ok             = checks["butterfly_ok"],
            calendar_ok              = checks["calendar_ok"],
            no_arbitrage             = checks["no_arbitrage"],
            max_cond1                = checks["max_cond1"],
            max_cond2                = checks["max_cond2"],
            pct_butterfly_violations = checks["pct_butterfly_violations"],
            n_calendar_violations    = checks["n_calendar_violations"],
            pct_calendar_violations  = checks["pct_calendar_violations"],
            worst_negative_diff      = checks["worst_negative_diff"],
        ))

        if verbose and (i + 1) % 50 == 0:
            print(f"  Checked {i + 1}/{len(ok)} dates …")

    df_out = pd.DataFrame(records)

    if verbose and len(df_out) > 0:
        pct_bf  = df_out["butterfly_ok"].mean() * 100
        pct_cal = df_out["calendar_ok"].mean() * 100
        pct_na  = df_out["no_arbitrage"].mean() * 100
        print(f"\nNo-arbitrage summary ({len(df_out)} dates):")
        print(f"  Butterfly-free : {pct_bf:.1f}%")
        print(f"  Calendar-free  : {pct_cal:.1f}%")
        print(f"  Fully arb-free : {pct_na:.1f}%")

    return df_out


def print_no_arbitrage_report(checks: dict) -> None:
    """Print a formatted no-arbitrage report for a single date."""
    sep    = "─" * 60
    status = "PASS" if checks["no_arbitrage"] else "FAIL"
    print(sep)
    print(f"  No-Arbitrage Check  [{status}]")
    print(sep)

    print(f"  Butterfly arbitrage : {'OK' if checks['butterfly_ok'] else 'VIOLATED'}")
    print(f"    max cond1  θφ(1+|ρ|)  = {checks['max_cond1']:.4f}  (limit 4.0)")
    print(f"    max cond2  θφ²(1+|ρ|) = {checks['max_cond2']:.4f}  (limit 4.0)")
    if not checks["butterfly_ok"]:
        print(f"    violations: {checks['n_butterfly_violations']} "
              f"({checks['pct_butterfly_violations']:.1f}% of T grid)")
    print()

    print(f"  Calendar arbitrage  : {'OK' if checks['calendar_ok'] else 'VIOLATED'}")
    print(f"    violations: {checks['n_calendar_violations']} "
          f"({checks['pct_calendar_violations']:.2f}% of grid)")
    if not checks["calendar_ok"]:
        print(f"    worst ΔW: {checks['worst_negative_diff']:.2e}")
    print(sep)


def plot_no_arbitrage_conditions(
    result: dict,
    T_grid: np.ndarray | None = None,
    n_T: int = 200,
    figsize: tuple = (11, 4),
) -> None:
    """Plot butterfly arbitrage conditions against the limit of 4.

    Left panel  : cond1 = theta * phi * (1 + |rho|)
    Right panel : cond2 = theta * phi^2 * (1 + |rho|)

    Red dashed line marks the limit of 4; violations are shaded.
    """
    bf     = check_butterfly_arbitrage(result, T_grid=T_grid, n_T=n_T)
    T_days = bf["T_grid"] * 365

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    for ax, cond, label in zip(
        axes,
        [bf["cond1"], bf["cond2"]],
        ["θφ(1+|ρ|)  [cond1]", "θφ²(1+|ρ|)  [cond2]"],
    ):
        ax.plot(T_days, cond, lw=2, color="steelblue")
        ax.axhline(4.0, color="crimson", lw=1.5, ls="--", label="Limit = 4")
        ax.fill_between(
            T_days, cond, 4.0,
            where=(cond >= 4.0), alpha=0.25, color="crimson", label="Violation",
        )
        ax.set_xlabel("TTE (days)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    status = "PASS" if bf["butterfly_ok"] else "FAIL"
    fig.suptitle(
        f"Butterfly Arbitrage Conditions — date {result.get('time_elapsed', '?')}  [{status}]",
        fontsize=12,
    )
    plt.show()


def plot_calendar_heatmap(
    result: dict,
    k_grid: np.ndarray | None = None,
    T_grid: np.ndarray | None = None,
    n_k: int = 100,
    n_T: int = 100,
    tolerance: float = 1e-8,
    figsize: tuple = (14, 5),
) -> None:
    """Plot calendar arbitrage diagnostic: ΔW heatmap and w(k,T) slices.

    Left panel  : heatmap of ΔW = w(k, T_{i+1}) − w(k, T_i); red = violation
    Right panel : w(k, T) curves for 6 selected maturities
    """
    cal    = check_calendar_arbitrage(result, k_grid=k_grid, T_grid=T_grid,
                                      n_k=n_k, n_T=n_T, tolerance=tolerance)
    k_grid = cal["k_grid"]
    T_grid = cal["T_grid"]
    dW     = cal["dw"]

    alpha = result["alpha"]
    beta  = result["beta"]
    rho   = result["rho"]
    eta   = result["eta"]
    gamma = result["gamma"]

    T_mid = 0.5 * (T_grid[:-1] + T_grid[1:]) * 365

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # ── Left: ΔW heatmap ────────────────────────────────────────────────────
    ax   = axes[0]
    vabs = max(float(np.abs(dW).max()), 1e-8)
    im   = ax.pcolormesh(
        T_mid, k_grid, dW,
        cmap="RdYlGn", vmin=-vabs, vmax=vabs, shading="auto",
    )
    plt.colorbar(im, ax=ax, label="ΔW = w(T+1) − w(T)")
    ax.set_xlabel("TTE (days, midpoint)")
    ax.set_ylabel("log(K/F)")
    cal_status = "PASS" if cal["calendar_ok"] else "FAIL"
    ax.set_title(f"Calendar Arbitrage ΔW  [{cal_status}]")

    if not cal["calendar_ok"]:
        bad_k_idx, bad_t_idx = np.where(dW < -tolerance)
        ax.scatter(
            T_mid[bad_t_idx], k_grid[bad_k_idx],
            s=8, color="crimson", zorder=5, label="Violations",
        )
        ax.legend(fontsize=9)

    # ── Right: w(k, T) slices ───────────────────────────────────────────────
    ax       = axes[1]
    n_slices = min(6, len(T_grid))
    T_idx    = np.linspace(0, len(T_grid) - 1, n_slices, dtype=int)
    colors   = plt.cm.plasma(np.linspace(0.1, 0.9, n_slices))

    for idx, color in zip(T_idx, colors):
        T_val   = T_grid[idx]
        w_slice = ssvi_total_variance_surface(
            k_grid, T_val, alpha, beta, rho, eta, gamma
        )
        ax.plot(k_grid, w_slice, color=color, lw=1.8, label=f"{T_val * 365:.0f}d")

    ax.set_xlabel("log(K/F)")
    ax.set_ylabel("Total variance w(k, T)")
    ax.set_title("w(k, T) Slices by Maturity")
    ax.legend(fontsize=8, title="TTE", ncol=2)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Calendar Arbitrage — date {result.get('time_elapsed', '?')}",
        fontsize=12,
    )
    plt.show()
