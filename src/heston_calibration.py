"""
heston_calibration.py
=====================
Heston model calibration — robust, IV-quality fitting.

Residual modes
--------------
price           : (C_model - C_market)                  [raw SPX points]
relative_price  : (C_model - C_market) / max(C_market, 1)   [default]
vega            : (C_model - C_market) / BlackVega       [IV-weighted]

Feller soft penalty (appended to residual vector)
    pen = lambda_feller * max(0, sigma_v² - 2·kappa·theta)

IV computation in calibrate_heston_all_dates is disabled by default
(~93 ms/option; set compute_iv=True only for diagnostic slices).
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm

from heston_pricing import heston_price_forward, batch_heston_prices, heston_to_black_iv
from forward_curve import interpolate_rate


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BOUNDS_LO = [1e-6, 1e-6,  0.01, 0.01, -0.999]
_BOUNDS_HI = [1.0,  1.0,  20.0,  5.0,   0.999]
_PARAMS    = ["v0", "theta", "kappa", "sigma_v", "rho"]


# ---------------------------------------------------------------------------
# 0. Black Vega helper (vectorised)
# ---------------------------------------------------------------------------

def _black_vega_batch(df_slice: pd.DataFrame) -> np.ndarray:
    """Black-forward vega DF·F·N'(d1)·√T for each option in the slice."""
    F  = df_slice["forward"].to_numpy(float)
    K  = df_slice["OPT STRIKE PRICE"].to_numpy(float)
    T  = df_slice["TTE"].to_numpy(float)
    r  = (df_slice["rate"].to_numpy(float)
          if "rate" in df_slice.columns else np.zeros(len(df_slice)))
    iv = (df_slice["implied_vol_forward"].to_numpy(float)
          if "implied_vol_forward" in df_slice.columns
          else np.full(len(df_slice), 0.20))

    iv = np.where((iv <= 0) | ~np.isfinite(iv), 0.20, iv)
    T  = np.maximum(T, 1e-6)
    df_val = np.exp(-r * T)
    sig_sqrtT = np.maximum(iv * np.sqrt(T), 1e-8)
    d1 = (np.log(np.maximum(F / K, 1e-9)) + 0.5 * iv**2 * T) / sig_sqrtT
    vega = df_val * F * norm.pdf(d1) * np.sqrt(T)
    return np.maximum(vega, 1e-4)


# ---------------------------------------------------------------------------
# 1. Slice preparation
# ---------------------------------------------------------------------------

def prepare_heston_slice(
    df: pd.DataFrame,
    time_elapsed: float,
    min_obs: int = 50,
    min_tte: float = 14 / 365,
    max_liquidity: float = 0.20,
    min_moneyness: float = -0.30,
    max_moneyness: float = 0.30,
    min_iv: float = 0.03,
    max_iv: float = 2.0,
) -> tuple[pd.DataFrame | None, str]:
    """Filter a single-date slice for Heston calibration.

    Builds a `rate` column from the term-structure if not already present.
    """
    sl = df[df["Time Elapsed"] == time_elapsed].copy()
    if len(sl) == 0:
        return None, f"No data for Time Elapsed = {time_elapsed}"

    req_cols = [
        "forward", "OPT STRIKE PRICE", "TTE",
        "implied_vol_forward", "forward_moneyness", "Mid Price",
        "Liquidity Factor", "OptionType",
    ]
    missing = [c for c in req_cols if c not in sl.columns]
    if missing:
        return None, f"Missing columns: {missing}"

    if "rate" not in sl.columns:
        sl["rate"] = sl.apply(interpolate_rate, axis=1)
    if sl["rate"].isna().all():
        return None, "Failed to interpolate rates from term structure"

    check = req_cols + ["rate"]
    sl = sl.dropna(subset=check)
    sl = sl[np.isfinite(sl[check]).all(axis=1)].copy()
    sl = sl[(sl["forward"] > 0) & (sl["OPT STRIKE PRICE"] > 0) & (sl["Mid Price"] > 0)].copy()
    sl = sl[(sl["rate"] > -0.5) & (sl["rate"] < 1.0)].copy()
    sl = sl[sl["TTE"] > min_tte].copy()
    sl = sl[(sl["implied_vol_forward"] >= min_iv) & (sl["implied_vol_forward"] <= max_iv)].copy()
    sl = sl[
        (sl["forward_moneyness"] >= min_moneyness)
        & (sl["forward_moneyness"] <= max_moneyness)
    ].copy()
    sl = sl[sl["Liquidity Factor"] <= max_liquidity].copy()

    if len(sl) < min_obs:
        return None, f"Only {len(sl)} observations (need {min_obs})"

    return sl.reset_index(drop=True), f"OK: {len(sl)} observations"


# ---------------------------------------------------------------------------
# 2. Initial guess
# ---------------------------------------------------------------------------

def initial_guess_heston(slice_df: pd.DataFrame) -> np.ndarray:
    """Data-driven initial guess.

    Distinguishes short-maturity (T < 0.5 yr) from long-maturity ATM IV
    to set v0 ≠ theta when the term structure is informative.
    """
    atm_mask = slice_df["forward_moneyness"].abs() <= 0.05
    atm      = slice_df.loc[atm_mask] if atm_mask.sum() > 0 else slice_df

    overall_iv = atm["implied_vol_forward"].median()
    v0_guess   = theta_guess = float(max(overall_iv**2, 1e-4))

    if "TTE" in atm.columns:
        short = atm[atm["TTE"] < 0.5]
        long_ = atm[atm["TTE"] >= 0.5]
        if len(short) > 0:
            v0_guess    = float(max(short["implied_vol_forward"].median()**2, 1e-4))
        if len(long_) > 0:
            theta_guess = float(max(long_["implied_vol_forward"].median()**2, 1e-4))

    return np.array([v0_guess, theta_guess, 2.0, 0.5, -0.7])


# ---------------------------------------------------------------------------
# 3. Residual function
# ---------------------------------------------------------------------------

def heston_residuals(
    params: np.ndarray,
    df_slice: pd.DataFrame,
    objective: str = "relative_price",
    lambda_feller: float = 5.0,
) -> np.ndarray:
    """Residuals for least_squares optimiser.

    Parameters
    ----------
    params        : [v0, theta, kappa, sigma_v, rho]
    df_slice      : clean options slice with rate column
    objective     : 'price' | 'relative_price' | 'vega'
    lambda_feller : weight on the Feller soft penalty

    Returns
    -------
    1D float array: option residuals + one Feller penalty element
    """
    v0, theta, kappa, sigma_v, rho = params

    prices_model  = batch_heston_prices(df_slice, v0, theta, kappa, sigma_v, rho)
    prices_market = df_slice["Mid Price"].to_numpy(float)

    # NaN guard: replace failed model prices with 10× market price
    # (large residual signals the optimiser away from the failure region)
    nan_mask      = np.isnan(prices_model)
    prices_model  = np.where(nan_mask, prices_market * 10.0, prices_model)

    raw_res = prices_model - prices_market

    if objective == "price":
        res = raw_res
    elif objective == "relative_price":
        denom = np.maximum(prices_market, 1.0)
        res   = raw_res / denom
    elif objective == "vega":
        vega = _black_vega_batch(df_slice)
        res  = raw_res / vega
    else:
        raise ValueError(
            f"Unknown objective '{objective}'. Use 'price', 'relative_price', or 'vega'."
        )

    # Feller soft penalty: max(0, sigma_v² - 2κθ)
    feller_gap = float(sigma_v**2 - 2.0 * kappa * theta)
    penalty    = lambda_feller * max(0.0, feller_gap)
    return np.append(res, penalty)


# ---------------------------------------------------------------------------
# 4. Single-slice calibration
# ---------------------------------------------------------------------------

def calibrate_heston_slice(
    df: pd.DataFrame,
    time_elapsed: float,
    objective: str = "relative_price",
    lambda_feller: float = 5.0,
    min_obs: int = 50,
    compute_iv: bool = False,
) -> dict:
    """Calibrate Heston model on a single date.

    Returns a dictionary with parameters, fit metrics, and diagnostics.
    Set compute_iv=True only for single-slice diagnostics (slow: ~93 ms/option).
    """
    base = dict(
        time_elapsed=time_elapsed,
        **{p: np.nan for p in _PARAMS},
        success=False, cost=np.nan, n_obs=0,
        rmse_price=np.nan, mae_price=np.nan,
        rmse_relative=np.nan, rmse_iv=np.nan,
        n_nan_prices=0, pct_nan_prices=0.0,
        feller_ok=True, feller_gap=np.nan,
        message="Not attempted",
    )

    sl, msg = prepare_heston_slice(df, time_elapsed, min_obs=min_obs)
    if sl is None:
        base["message"] = msg
        return base

    base["n_obs"] = len(sl)
    x0 = initial_guess_heston(sl)

    try:
        opt = least_squares(
            heston_residuals,
            x0,
            args=(sl, objective, lambda_feller),
            bounds=(_BOUNDS_LO, _BOUNDS_HI),
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
            max_nfev=5_000,
        )
    except Exception as exc:
        base["message"] = f"Optimiser error: {exc}"
        return base

    v0_h, theta_h, kappa_h, sigma_v_h, rho_h = opt.x

    # Price diagnostics
    prices_model  = batch_heston_prices(sl, v0_h, theta_h, kappa_h, sigma_v_h, rho_h)
    prices_market = sl["Mid Price"].to_numpy(float)
    n_nan         = int(np.sum(np.isnan(prices_model)))
    pm_safe       = np.where(np.isnan(prices_model), prices_market, prices_model)

    rmse_price    = float(np.sqrt(np.mean((pm_safe - prices_market) ** 2)))
    mae_price     = float(np.mean(np.abs(pm_safe - prices_market)))
    rmse_relative = float(np.sqrt(np.mean(
        ((pm_safe - prices_market) / np.maximum(prices_market, 1.0)) ** 2
    )))

    # IV diagnostics (optional — slow)
    rmse_iv = np.nan
    if compute_iv and "implied_vol_forward" in sl.columns:
        iv_model = np.array([
            heston_to_black_iv(
                pm_safe[i],
                float(sl["forward"].iloc[i]),
                float(sl["OPT STRIKE PRICE"].iloc[i]),
                float(sl["TTE"].iloc[i]),
                float(sl["rate"].iloc[i]),
                int(sl["OptionType"].iloc[i]),
            )
            for i in range(len(sl))
        ])
        iv_market = sl["implied_vol_forward"].to_numpy(float)
        valid = np.isfinite(iv_model) & np.isfinite(iv_market)
        if valid.sum() > 10:
            rmse_iv = float(np.sqrt(np.mean((iv_model[valid] - iv_market[valid]) ** 2)))

    feller_ok  = 2.0 * kappa_h * theta_h >= 0.99 * sigma_v_h ** 2
    feller_gap = float(sigma_v_h ** 2 - 2.0 * kappa_h * theta_h)

    return dict(
        time_elapsed=time_elapsed,
        v0=float(v0_h), theta=float(theta_h), kappa=float(kappa_h),
        sigma_v=float(sigma_v_h), rho=float(rho_h),
        success=bool(opt.success), cost=float(opt.cost),
        n_obs=len(sl),
        rmse_price=rmse_price, mae_price=mae_price,
        rmse_relative=rmse_relative, rmse_iv=rmse_iv,
        n_nan_prices=n_nan, pct_nan_prices=100.0 * n_nan / len(sl),
        feller_ok=feller_ok, feller_gap=feller_gap,
        message=opt.message,
    )


# ---------------------------------------------------------------------------
# 5. Fit quality on a single slice
# ---------------------------------------------------------------------------

def compute_heston_fit(df_slice: pd.DataFrame, result: dict) -> pd.DataFrame:
    """Add Heston prices and (optionally) implied vols to a slice.

    Returns DataFrame with new columns:
        heston_price, price_error, heston_iv (if implied_vol_forward present), iv_error
    """
    v0      = result["v0"]
    theta   = result["theta"]
    kappa   = result["kappa"]
    sigma_v = result["sigma_v"]
    rho     = result["rho"]

    df = df_slice.copy()
    df["heston_price"] = batch_heston_prices(df, v0, theta, kappa, sigma_v, rho)
    df["price_error"]  = df["heston_price"] - df["Mid Price"]

    df["heston_iv"] = df.apply(
        lambda row: heston_to_black_iv(
            row["heston_price"],
            row["forward"],
            row["OPT STRIKE PRICE"],
            row["TTE"],
            row["rate"],
            int(row["OptionType"]),
        ),
        axis=1,
    )

    if "implied_vol_forward" in df.columns:
        df["iv_error"] = df["heston_iv"] - df["implied_vol_forward"]

    return df


# ---------------------------------------------------------------------------
# 6. Human-readable diagnostics
# ---------------------------------------------------------------------------

def print_heston_diagnostics(result: dict) -> None:
    """Print structured diagnostics for a single calibration result."""
    SEP = "=" * 58
    print(f"\n{SEP}")
    print(f"  Heston Calibration   Time Elapsed = {result.get('time_elapsed', '?')}")
    print(SEP)

    v0      = result.get("v0",      np.nan)
    theta   = result.get("theta",   np.nan)
    kappa   = result.get("kappa",   np.nan)
    sigma_v = result.get("sigma_v", np.nan)
    rho     = result.get("rho",     np.nan)

    print("\nParameters:")
    print(f"  v0        = {v0:.6f}   instant variance")
    print(f"  theta     = {theta:.6f}   long-run variance")
    print(f"  kappa     = {kappa:.4f}   mean-reversion speed")
    print(f"  sigma_v   = {sigma_v:.4f}   vol of vol")
    print(f"  rho       = {rho:.4f}   leverage correlation")

    print("\nEconomic interpretation:")
    if np.isfinite(v0):
        print(f"  Instant vol  (√v0)    = {100*np.sqrt(v0):.2f}%")
    if np.isfinite(theta):
        print(f"  Long-run vol (√θ)     = {100*np.sqrt(theta):.2f}%")
    if np.isfinite(kappa):
        hl_yr  = np.log(2) / kappa
        hl_day = hl_yr * 365
        print(f"  Half-life   ln2/κ     = {hl_yr:.3f} yr  ({hl_day:.0f} days)")

    feller_2kt = 2.0 * kappa * theta if np.isfinite(kappa) and np.isfinite(theta) else np.nan
    feller_gap = result.get("feller_gap", np.nan)
    feller_ok  = result.get("feller_ok",  None)
    print(f"\nFeller condition  2κθ ≥ σᵥ²:")
    if np.isfinite(feller_2kt):
        status = "OK" if feller_ok else "VIOLATED — variance can reach zero"
        print(f"  2κθ    = {feller_2kt:.6f}")
        print(f"  σᵥ²    = {sigma_v**2:.6f}")
        print(f"  gap    = {feller_gap:+.6f}  [{status}]")

    print("\nFit quality:")
    for label, key in [
        ("RMSE price    ", "rmse_price"),
        ("MAE  price    ", "mae_price"),
        ("RMSE relative ", "rmse_relative"),
        ("RMSE IV       ", "rmse_iv"),
    ]:
        val = result.get(key, np.nan)
        if np.isfinite(val):
            print(f"  {label} = {val:.6f}")

    n_nan = result.get("n_nan_prices", 0)
    pct   = result.get("pct_nan_prices", 0.0)
    print(f"\nData / convergence:")
    print(f"  n_obs       = {result.get('n_obs', '?')}")
    print(f"  NaN prices  = {n_nan} ({pct:.1f}%)")
    print(f"  success     = {result.get('success', '?')}")
    print(f"  message     = {result.get('message', '?')[:60]}")
    print(SEP + "\n")


# ---------------------------------------------------------------------------
# 7. Multi-date calibration
# ---------------------------------------------------------------------------

def calibrate_heston_all_dates(
    df: pd.DataFrame,
    dates: list | None = None,
    objective: str = "relative_price",
    lambda_feller: float = 5.0,
    min_obs: int = 50,
    compute_iv: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Calibrate Heston on all (or specified) dates.

    Parameters
    ----------
    compute_iv : set True only for short runs — IV inversion is ~93 ms/option
    """
    if dates is None:
        dates = sorted(df["Time Elapsed"].unique())

    n    = len(dates)
    rows = []

    for i, te in enumerate(dates, 1):
        result = calibrate_heston_slice(
            df, te,
            objective=objective,
            lambda_feller=lambda_feller,
            min_obs=min_obs,
            compute_iv=compute_iv,
        )
        rows.append(result)

        if verbose and (i == 1 or i % max(1, n // 10) == 0 or i == n):
            n_ok    = sum(r["success"] for r in rows)
            rmse_p  = result.get("rmse_price", float("nan"))
            feller  = "✓" if result.get("feller_ok", False) else "✗"
            print(
                f"  {i:>4}/{n}  success={n_ok}/{i:>3}  "
                f"rmse_price={rmse_p:.4f}  feller={feller}"
            )

    results_df = pd.DataFrame(rows)
    if verbose:
        n_ok = int(results_df["success"].sum())
        print(f"\nFinished: {n_ok}/{n} ({100*n_ok/n:.1f}%) calibrated successfully")
        if "feller_ok" in results_df.columns:
            n_fell = int(results_df["feller_ok"].sum())
            print(f"Feller OK: {n_fell}/{n_ok} successful calibrations")

    return results_df


# ---------------------------------------------------------------------------
# 8. Backward-compatible predict wrapper
# ---------------------------------------------------------------------------

def predict_heston_slice(
    df_slice: pd.DataFrame,
    result: dict,
    compute_iv: bool = False,
) -> pd.DataFrame:
    """Add Heston model predictions to a data slice.

    Kept for backward compatibility; delegates to compute_heston_fit when
    compute_iv=True.
    """
    v0      = result["v0"]
    theta   = result["theta"]
    kappa   = result["kappa"]
    sigma_v = result["sigma_v"]
    rho     = result["rho"]

    df = df_slice.copy()
    df["heston_price"]       = batch_heston_prices(df, v0, theta, kappa, sigma_v, rho)
    df["heston_price_error"] = df["heston_price"] - df["Mid Price"]

    if compute_iv:
        fit = compute_heston_fit(df_slice, result)
        df["heston_iv"] = fit["heston_iv"]
        if "iv_error" in fit.columns:
            df["iv_error"] = fit["iv_error"]

    return df
