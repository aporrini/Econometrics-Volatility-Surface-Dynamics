"""
heston_pricing.py
=================
Forward-based Heston pricing via P1/P2 Gil-Pelaez inversion.

Model (forward measure, zero drift):
    dF_t = sqrt(v_t) * F_t * dW_t
    dv_t = kappa*(theta - v_t)*dt + sigma_v*sqrt(v_t)*dW_t^v
    corr(dW_t, dW_t^v) = rho

European call: C = DF*(F*P1 - K*P2)
where P1, P2 are risk-neutral probabilities obtained via Gil-Pelaez inversion.
Lord-Kahl (2010) sign fix enforces Re(d) >= 0 to prevent branch-cut errors.
"""

import numpy as np
import pandas as pd

_PI     = np.pi
_N_PTS  = 500
_U_MAX  = 400.0
_U_GRID = np.linspace(1e-5, _U_MAX, _N_PTS)


# ---------------------------------------------------------------------------
# 1. Characteristic function (Lord-Kahl stable, vectorized over u)
# ---------------------------------------------------------------------------

def _cf(
    u: np.ndarray,
    T: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
) -> np.ndarray:
    """Characteristic function of log(F_T/F_0) — Lord-Kahl stable form.

    Derivation: Feynman-Kac on dx = -v/2 dt + sqrt(v) dW gives Riccati ODE
    whose solution is:
        phi(u) = exp(C(u,T)*theta + D(u,T)*v0)
        alpha  = kappa - i*rho*sigma_v*u
        d      = sqrt(alpha^2 + sigma_v^2*(u^2 + i*u))
        g      = (alpha - d) / (alpha + d)
        C      = (kappa/sigma_v^2) * [(alpha-d)*T - 2*log((1-g*e^{-dT})/(1-g))]
        D      = (alpha-d)/sigma_v^2 * (1-e^{-dT}) / (1-g*e^{-dT})

    The Lord-Kahl fix d -> -d when Re(d)<0 keeps exp(-dT) decaying and
    avoids the log branch cut discontinuity ("Heston trap").
    """
    s2    = sigma_v ** 2
    alpha = kappa - 1j * rho * sigma_v * u
    d     = np.sqrt(alpha ** 2 + s2 * (u ** 2 + 1j * u))
    d     = np.where(np.real(d) < 0.0, -d, d)      # Lord-Kahl sign fix
    g     = (alpha - d) / (alpha + d)
    edT   = np.exp(-d * T)
    logQ  = np.log((1.0 - g * edT) / (1.0 - g))
    C     = (kappa * theta / s2) * ((alpha - d) * T - 2.0 * logQ)
    D     = ((alpha - d) / s2) * (1.0 - edT) / (1.0 - g * edT)
    return np.exp(C + D * v0)


# ---------------------------------------------------------------------------
# 2. Call price via Gil-Pelaez P1/P2 inversion
# ---------------------------------------------------------------------------

def heston_call_price_forward(
    F: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    theta: float,
    kappa: float,
    sigma_v: float,
    rho: float,
) -> float:
    """European call under Heston (forward measure).

    C = DF * (F*P1 - K*P2)

    P2 = 1/2 + (1/pi) * int_0^inf Re[ e^{-iuk} * phi(u)   / (iu) ] du
    P1 = 1/2 + (1/pi) * int_0^inf Re[ e^{-iuk} * phi(u-i) / (iu) ] du

    where k = log(K/F) is log-moneyness and phi is the CF of log(F_T/F_0).
    phi(-i) = E[F_T/F_0] = 1 under the forward measure, so no normalisation needed.

    Numerical integration via vectorised trapezoidal rule on [1e-5, 400].
    """
    if not np.isfinite([F, K, T, r, v0, theta, kappa, sigma_v, rho]).all():
        return np.nan
    if F <= 0 or K <= 0 or T <= 0 or v0 <= 0 or theta <= 0 or kappa <= 0 or sigma_v <= 0:
        return np.nan
    if abs(rho) >= 1.0:
        return np.nan

    df = np.exp(-r * T)
    k  = np.log(K / F)
    u  = _U_GRID

    try:
        ek    = np.exp(-1j * u * k)
        phi2  = _cf(u,       T, v0, kappa, theta, sigma_v, rho)
        phi1  = _cf(u - 1j,  T, v0, kappa, theta, sigma_v, rho)

        # Re[phi / (i*u)] = Im[phi] / u  (Gil-Pelaez)
        intg2 = np.nan_to_num(np.real(ek * phi2 / (1j * u)))
        intg1 = np.nan_to_num(np.real(ek * phi1 / (1j * u)))

        I2 = np.trapz(intg2, u)
        I1 = np.trapz(intg1, u)
    except Exception:
        return np.nan

    P2   = float(np.clip(0.5 + I2 / _PI, 0.0, 1.0))
    P1   = float(np.clip(0.5 + I1 / _PI, 0.0, 1.0))
    call = df * (F * P1 - K * P2)
    return float(np.clip(call, 0.0, df * F))


# ---------------------------------------------------------------------------
# 3. Put via put-call parity
# ---------------------------------------------------------------------------

def heston_put_price_forward(
    F: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    theta: float,
    kappa: float,
    sigma_v: float,
    rho: float,
) -> float:
    """European put via put-call parity: P = C - DF*(F - K)."""
    if not np.isfinite([F, K, T, r, v0, theta, kappa, sigma_v, rho]).all():
        return np.nan
    if F <= 0 or K <= 0 or T <= 0:
        return np.nan
    call = heston_call_price_forward(F, K, T, r, v0, theta, kappa, sigma_v, rho)
    if not np.isfinite(call):
        return np.nan
    df  = np.exp(-r * T)
    return float(np.clip(call - df * (F - K), 0.0, df * K))


# ---------------------------------------------------------------------------
# 4. Generic dispatcher
# ---------------------------------------------------------------------------

def heston_price_forward(
    F: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    theta: float,
    kappa: float,
    sigma_v: float,
    rho: float,
    option_type: int,
) -> float:
    """Price European option (call=1, put=-1) under Heston forward model."""
    if option_type == 1:
        return heston_call_price_forward(F, K, T, r, v0, theta, kappa, sigma_v, rho)
    if option_type == -1:
        return heston_put_price_forward(F, K, T, r, v0, theta, kappa, sigma_v, rho)
    return np.nan


# ---------------------------------------------------------------------------
# 5. Batch pricing
# ---------------------------------------------------------------------------

def batch_heston_prices(
    df: pd.DataFrame,
    v0: float,
    theta: float,
    kappa: float,
    sigma_v: float,
    rho: float,
) -> np.ndarray:
    """Compute Heston prices for a batch of options.

    Parameters
    ----------
    df : DataFrame with columns:
        forward, OPT STRIKE PRICE, TTE, rate (or built from term structure), OptionType
    v0, theta, kappa, sigma_v, rho : Heston parameters

    Returns
    -------
    1D float array of model prices (NaN for invalid rows)
    """
    df_c = df.copy()
    if "rate" not in df_c.columns:
        from forward_curve import interpolate_rate
        df_c["rate"] = df_c.apply(interpolate_rate, axis=1)

    prices = np.full(len(df_c), np.nan, dtype=float)
    for idx, (_, row) in enumerate(df_c.iterrows()):
        F = float(row.get("forward",          np.nan))
        K = float(row.get("OPT STRIKE PRICE", np.nan))
        T = float(row.get("TTE",              np.nan))
        r = float(row.get("rate",             np.nan))
        ot = row.get("OptionType", np.nan)
        if not np.isfinite([F, K, T, r]).all():
            continue
        try:
            opt_type = int(ot)
        except (ValueError, TypeError):
            continue
        prices[idx] = heston_price_forward(F, K, T, r, v0, theta, kappa, sigma_v, rho, opt_type)

    return prices


# ---------------------------------------------------------------------------
# 6. Black implied volatility from Heston price (diagnostic utility)
# ---------------------------------------------------------------------------

def heston_to_black_iv(
    heston_price: float,
    F: float,
    K: float,
    T: float,
    r: float,
    option_type: int,
    sigma_lo: float = 1e-6,
    sigma_hi: float = 5.0,
) -> float:
    """Invert Black forward model to recover IV from a Heston price.

    SLOW: use only for diagnostics, not per-option in calibration.
    Returns np.nan if inversion fails.
    """
    from black_scholes import black_forward_price
    from scipy.optimize import brentq

    if not np.isfinite([heston_price, F, K, T, r]).all():
        return np.nan
    if heston_price <= 0 or F <= 0 or K <= 0 or T <= 0:
        return np.nan

    def objective(sigma):
        return black_forward_price(F, K, T, r, sigma, option_type) - heston_price

    try:
        f_lo = objective(sigma_lo)
        f_hi = objective(sigma_hi)
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
            return np.nan
        return float(brentq(objective, sigma_lo, sigma_hi, xtol=1e-8, maxiter=200))
    except (ValueError, RuntimeError):
        return np.nan
