"""
black_scholes.py
================
Black (1976) forward-model pricing and implied volatility for European options.

C = DF * [F * N(d1) - K * N(d2)]
P = DF * [K * N(-d2) - F * N(-d1)]

where DF = exp(-r*T), d1 = [log(F/K) + 0.5*sigma^2*T] / (sigma*sqrt(T)), d2 = d1 - sigma*sqrt(T).

Required DataFrame columns:
    forward          – implied forward F(date, TTE)  [from forward_curve.py]
    OPT STRIKE PRICE – strike K
    TTE              – time to expiry in years
    Mid Price        – observed mid price
    OptionType       – 1 = call, -1 = put
    rate1month … rate3year – term-structure rates (percentage form, e.g. 1.20 = 1.20%)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Rate interpolation (local helper)
# ---------------------------------------------------------------------------

_RATE_MATURITIES = np.array([1/12, 3/12, 6/12, 1.0, 2.0, 3.0])
_RATE_COLS = ["rate1month", "rate3month", "rate6month", "rate1year", "rate2year", "rate3year"]


def interpolate_rate(row) -> float:
    """Linearly interpolate risk-free rate from the term structure for one row."""
    T = row["TTE"]
    if not np.isfinite(T) or T <= 0:
        return np.nan
    rates = np.array([row[c] for c in _RATE_COLS], dtype=float)
    if np.any(~np.isfinite(rates)):
        return np.nan
    return float(np.interp(T, _RATE_MATURITIES, rates / 100.0))


# ---------------------------------------------------------------------------
# Black (1976) pricing
# ---------------------------------------------------------------------------

def black_forward_price(
    F: float, K: float, T: float, r: float, sigma: float, option_type: int,
) -> float:
    """Black (1976) price of a European option on a forward.

    Returns np.nan for invalid inputs (non-finite, non-positive, or wrong option_type).
    """
    if option_type not in (1, -1):
        return np.nan
    if any(not np.isfinite(x) for x in (F, K, T, r, sigma)):
        return np.nan
    if F <= 0 or K <= 0:
        return np.nan

    DF = np.exp(-r * T)

    if T <= 0 or sigma <= 0:
        if option_type == 1:
            return DF * max(F - K, 0.0)
        return DF * max(K - F, 0.0)

    sqrt_T = np.sqrt(T)
    d1     = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt_T)
    d2     = d1 - sigma * sqrt_T

    if option_type == 1:
        return DF * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return DF * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


# ---------------------------------------------------------------------------
# Implied volatility (forward model)
# ---------------------------------------------------------------------------

def implied_vol_forward(
    mid_price: float, F: float, K: float, T: float, r: float, option_type: int,
    sigma_lo: float = 1e-6, sigma_hi: float = 5.0,
) -> float:
    """Invert the Black (1976) formula via Brent's method.

    Returns np.nan when the problem is ill-posed.
    """
    if option_type not in (1, -1):
        return np.nan
    if any(not np.isfinite(x) for x in (mid_price, F, K, T, r)):
        return np.nan
    if mid_price <= 0 or F <= 0 or K <= 0 or T <= 0:
        return np.nan

    DF        = np.exp(-r * T)
    intrinsic = DF * max(option_type * (F - K), 0.0)
    if mid_price < intrinsic:
        return np.nan

    def objective(sigma: float) -> float:
        return black_forward_price(F, K, T, r, sigma, option_type) - mid_price

    try:
        f_lo = objective(sigma_lo)
        f_hi = objective(sigma_hi)
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
            return np.nan
        if f_lo * f_hi > 0:
            return np.nan
        return float(brentq(objective, sigma_lo, sigma_hi, xtol=1e-8, maxiter=200))
    except (ValueError, RuntimeError, OverflowError):
        return np.nan


def implied_vol_forward_from_row(row, sigma_lo: float = 1e-6, sigma_hi: float = 5.0) -> float:
    """Compute Black forward IV for one DataFrame row.

    Requires a 'forward' column (from attach_forward_to_options).
    Interpolates the risk-free rate from the term-structure columns.
    """
    r = interpolate_rate(row)
    if not np.isfinite(r):
        return np.nan
    F = float(row.get("forward", np.nan))
    if not (np.isfinite(F) and F > 0):
        return np.nan
    return implied_vol_forward(
        mid_price   = float(row["Mid Price"]),
        F           = F,
        K           = float(row["OPT STRIKE PRICE"]),
        T           = float(row["TTE"]),
        r           = r,
        option_type = int(row["OptionType"]),
        sigma_lo    = sigma_lo,
        sigma_hi    = sigma_hi,
    )


# ---------------------------------------------------------------------------
# Vectorized helpers for fast IV computation
# ---------------------------------------------------------------------------

def _interp_rates_vec(T: np.ndarray, rates_pct: np.ndarray) -> np.ndarray:
    """Vectorized piecewise-linear rate interpolation (one rate per row).

    Parameters
    ----------
    T         : (N,) TTE in years
    rates_pct : (N, 6) rate columns in percent (e.g. 1.20 means 1.20%)

    Returns
    -------
    r : (N,) risk-free rates in decimal
    """
    xp   = _RATE_MATURITIES
    fp   = rates_pct / 100.0
    idx  = np.clip(np.searchsorted(xp, T, side="right") - 1, 0, len(xp) - 2)
    rows = np.arange(len(T))
    x0, x1 = xp[idx], xp[idx + 1]
    y0, y1  = fp[rows, idx], fp[rows, idx + 1]
    w    = np.where(x1 > x0, np.clip((T - x0) / (x1 - x0), 0.0, 1.0), 0.0)
    r    = y0 + w * (y1 - y0)
    bad  = ~np.isfinite(T) | (T <= 0) | np.any(~np.isfinite(rates_pct), axis=1)
    r[bad] = np.nan
    return r


def _black76_price_vega_vec(
    F: np.ndarray, K: np.ndarray, T: np.ndarray,
    r: np.ndarray, sigma: np.ndarray, otype: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Black-76 price and vega on full arrays."""
    DF    = np.exp(-r * T)
    sqrtT = np.sqrt(np.maximum(T, 1e-12))
    sig   = np.maximum(sigma, 1e-9)
    logFK = np.log(np.where((F > 0) & (K > 0), F / K, 1.0))
    d1    = (logFK + 0.5 * sig ** 2 * T) / (sig * sqrtT)
    d2    = d1 - sig * sqrtT
    Nd1, Nd2 = norm.cdf(d1), norm.cdf(d2)
    nd1   = norm.pdf(d1)
    price = np.where(
        otype == 1,
        DF * (F * Nd1 - K * Nd2),
        DF * (K * (1.0 - Nd2) - F * (1.0 - Nd1)),
    )
    vega  = DF * F * sqrtT * nd1
    return price, vega


def _iv_nr_vec(
    mid: np.ndarray, F: np.ndarray, K: np.ndarray,
    T: np.ndarray, r: np.ndarray, otype: np.ndarray,
    sigma_lo: float = 1e-6, sigma_hi: float = 5.0,
    n_iter: int = 15, tol: float = 1e-5,
) -> np.ndarray:
    """Vectorized Newton-Raphson Black-76 IV inversion.

    Returns implied vol for each row; np.nan where no solution exists.
    Starting guess: Brenner-Subrahmanyam ATM approximation, clipped to [sigma_lo, sigma_hi].
    """
    valid = (
        np.isfinite(mid) & np.isfinite(F) & np.isfinite(K) &
        np.isfinite(T)   & np.isfinite(r) &
        (mid > 0) & (F > 0) & (K > 0) & (T > 0)
    )
    T_safe = np.where(valid, T, 1.0)
    r_safe = np.where(valid, r, 0.0)
    DF        = np.exp(-r_safe * T_safe)
    intrinsic = DF * np.maximum(otype * (F - K), 0.0)
    valid    &= mid >= intrinsic - 1e-8

    # Brenner-Subrahmanyam initial guess (near-ATM approximation)
    denom = np.where(valid & (F * DF > 1e-8), F * DF, 1.0)
    sigma = np.where(
        valid,
        np.clip(np.sqrt(2.0 * np.pi / T_safe) * mid / denom, sigma_lo, sigma_hi),
        0.3,
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(n_iter):
            price, vega = _black76_price_vega_vec(F, K, T_safe, r_safe, sigma, otype)
            step  = np.where(valid & (vega > 1e-10), (price - mid) / vega, 0.0)
            sigma = np.clip(sigma - step, sigma_lo, sigma_hi)

    price_final, _ = _black76_price_vega_vec(F, K, T_safe, r_safe, sigma, otype)
    converged = valid & (np.abs(price_final - mid) < tol)
    return np.where(converged, sigma, np.nan)


# ---------------------------------------------------------------------------
# Public function (vectorized)
# ---------------------------------------------------------------------------

def add_implied_volatility_forward(
    df: pd.DataFrame,
    min_iv: float       = 0.03,
    max_iv: float       = 2.0,
    sigma_lo: float     = 1e-6,
    sigma_hi: float     = 5.0,
    report: list | None = None,
    n_iter: int         = 15,
    tol: float          = 1e-5,
) -> pd.DataFrame:
    """Compute Black (1976) forward IV and attach it to the DataFrame.

    Uses vectorized Newton-Raphson (15 iterations, Brenner-Subrahmanyam init).
    ~20-50x faster than row-by-row Brent on large datasets.

    Requires a 'forward' column from attach_forward_to_options().
    Rows without a valid solution or outside [min_iv, max_iv] are dropped.
    """
    if "forward" not in df.columns:
        raise KeyError("Column 'forward' not found — run attach_forward_to_options() first.")

    df = df.copy()
    df.columns = df.columns.str.strip()
    n_before = len(df)

    mid   = df["Mid Price"].to_numpy(float)
    F     = df["forward"].to_numpy(float)
    K     = df["OPT STRIKE PRICE"].to_numpy(float)
    T     = df["TTE"].to_numpy(float)
    otype = df["OptionType"].to_numpy(float)
    rates = df[_RATE_COLS].to_numpy(float)
    r     = _interp_rates_vec(T, rates)

    df["implied_vol_forward"] = _iv_nr_vec(
        mid, F, K, T, r, otype,
        sigma_lo=sigma_lo, sigma_hi=sigma_hi,
        n_iter=n_iter, tol=tol,
    )

    df        = df[np.isfinite(df["implied_vol_forward"])].copy()
    n_finite  = len(df)

    df        = df[(df["implied_vol_forward"] >= min_iv) & (df["implied_vol_forward"] <= max_iv)].copy()
    n_bounded = len(df)

    if report is not None:
        report.append({"step": "Forward IV non-finite (no solution / no forward)",
                       "n_before": n_before, "n_after": n_finite})
        report.append({"step": f"Forward IV outside [{min_iv:.2f}, {max_iv:.2f}]",
                       "n_before": n_finite,  "n_after": n_bounded})

    return df
