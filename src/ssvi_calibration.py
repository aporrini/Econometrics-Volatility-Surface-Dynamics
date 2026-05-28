"""
ssvi_calibration.py
===================
Per-bucket SSVI calibration for S&P 500 implied volatility surface.

Improvements over the previous version
---------------------------------------
- Calibration is performed **per maturity bucket** (short / medium / long),
  never across all maturities at once.  Mixing maturities forces phi to
  compensate for the term structure, driving it to its upper bound.

- The objective is minimised in **total-variance space** (w = sigma^2 * T),
  which is the natural space for SSVI and avoids the heteroskedastic residuals
  that arise when minimising on sigma directly.

- **Finance-consistent initial guess**: theta from ATM w, phi from the
  observed slope of w vs. k near ATM (dw/dk|0 = theta * rho * phi).

- **Tighter bounds**: phi <= 20 instead of 100.

- **Soft penalties** for phi > 10 and |rho| near 1, appended to the
  least-squares residual vector.

- **Robust cleaning** per slice: IV, moneyness, liquidity, and TTE filters
  applied before optimisation.

SSVI formula (Gatheral & Jacquier 2014)
----------------------------------------
    w(k; theta, rho, phi) = (theta/2) * [
        1 + rho * phi * k + sqrt((phi * k + rho)^2 + 1 - rho^2)
    ]

where w = sigma^2_IV * T is the total implied variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import norm


# ─── Maturity bucket boundaries ───────────────────────────────────────────────

# TTE in years.  Options with TTE < 20/365 are excluded from all buckets.
_BUCKET_LO: dict[str, float] = {
    "short":  20  / 365,
    "medium": 60  / 365,
    "long":   180 / 365,
}
_BUCKET_HI: dict[str, float] = {
    "short":  60  / 365,
    "medium": 180 / 365,
    "long":   np.inf,
}
BUCKET_NAMES: tuple[str, ...] = ("short", "medium", "long")

# ─── Optimiser bounds: (theta, rho, phi) ──────────────────────────────────────
_LO = np.array([1e-6, -0.999,  1e-4])
_HI = np.array([2.0,   0.999, 20.0])

# Soft-penalty strength added to the residual vector for extreme phi / rho
_PENALTY = 100.0

# ─── Delta filter parameters ──────────────────────────────────────────────────
SSVI_DELTA_SHORT: float = 0.30   # abs_delta lower bound as T → 0  (narrow band)
SSVI_DELTA_LONG:  float = 0.05   # abs_delta lower bound as T → ∞  (wide band)
SSVI_DELTA_DECAY: float = 3.0    # exponential decay rate (years⁻¹)

# ─── Balance enforcement parameters ──────────────────────────────────────────
SSVI_MIN_LEFT_POINTS:  int   = 5     # min OTM-put (k<0) observations after filter
SSVI_MIN_RIGHT_POINTS: int   = 5     # min OTM-call (k>0) observations after filter
SSVI_MIN_ATM_POINTS:   int   = 3     # min ATM observations after filter
SSVI_ATM_BAND:         float = 0.03  # |k| ≤ this → ATM zone
SSVI_FALLBACK_N:       int   = 10    # max fallback options added per side


def delta_min_tte(
    T: np.ndarray,
    delta_short: float = SSVI_DELTA_SHORT,
    delta_long:  float = SSVI_DELTA_LONG,
    decay:       float = SSVI_DELTA_DECAY,
) -> np.ndarray:
    """Continuous lower bound on abs_delta as a function of TTE.

        delta_min(T) = delta_long + (delta_short - delta_long) * exp(-decay * T)

    Limits
    ------
    T → 0 : delta_min → delta_short  (narrow band, exclude deep wings)
    T → ∞ : delta_min → delta_long   (wide band, include most of the surface)
    """
    T = np.asarray(T, dtype=float)
    return delta_long + (delta_short - delta_long) * np.exp(-decay * T)


def filter_delta_butterfly_continuous(
    df: pd.DataFrame,
    delta_short: float = SSVI_DELTA_SHORT,
    delta_long:  float = SSVI_DELTA_LONG,
    decay:       float = SSVI_DELTA_DECAY,
    *,
    min_left_points:  int   = SSVI_MIN_LEFT_POINTS,
    min_right_points: int   = SSVI_MIN_RIGHT_POINTS,
    min_atm_points:   int   = SSVI_MIN_ATM_POINTS,
    atm_band:         float = SSVI_ATM_BAND,
    n_fallback:       int   = SSVI_FALLBACK_N,
) -> pd.DataFrame:
    """Side-aware delta butterfly filter with automatic balance enforcement.

    The lower-delta bound decays smoothly with TTE:

        delta_min(T) = delta_long + (delta_short - delta_long) * exp(-decay * T)

    Initial keep condition:

        delta_min(T)  <=  abs_delta  <=  1 - delta_min(T)

    Balance enforcement
    -------------------
    After the initial filter, if either side of the smile has fewer than the
    minimum required points, the function adds back the discarded options
    closest to ATM (highest abs_delta) from that side — up to ``n_fallback``
    options per side.  This ensures both wings constrain the calibration.

    Smile-side classification (based on forward log-moneyness k = log(K/F))
    ------------------------------------------------------------------------
    left_put_otm   : k < -atm_band   (OTM puts, left wing)
    atm            : |k| ≤ atm_band  (near-money options)
    right_call_otm : k >  atm_band   (OTM calls, right wing)

    Forward Black-Scholes delta:

        d1        = (log(F/K) + 0.5·σ²·T) / (σ·√T)
        call_delta = N(d1)
        abs_delta  = call_delta       for calls  (OptionType == 1)
                   = |call_delta − 1| for puts   (OptionType == -1)

    Column auto-detection
    ---------------------
    Moneyness : prefers ``forward_moneyness`` = log(K/F); falls back to
                ``Moneyness`` = log(K/S).
    IV        : prefers ``implied_vol_forward``; falls back to ``implied_vol``.
    OptionType: if absent, uses OTM convention (k≥0 → call, k<0 → put).

    Added columns
    -------------
    abs_delta           : forward Black delta in absolute terms
    delta_min_tte       : lower bound delta_min(T) for each row
    keep_ssvi           : True if the row is included (initial filter or fallback)
    smile_side          : 'left_put_otm' | 'atm' | 'right_call_otm'

    DataFrame.attrs keys
    --------------------
    n_left_put_otm       : OTM-put observations kept (left wing)
    n_right_call_otm     : OTM-call observations kept (right wing)
    n_atm                : ATM observations kept
    n_total_after_filter : total kept
    flag_unbalanced_smile: True if either wing is still empty after fallback

    Returns
    -------
    Filtered copy of df (kept rows) with the four added columns.
    """
    import warnings as _warnings

    df = df.copy().reset_index(drop=True)

    k_col  = "forward_moneyness"   if "forward_moneyness"   in df.columns else "Moneyness"
    iv_col = "implied_vol_forward" if "implied_vol_forward" in df.columns else "implied_vol"

    for col in (k_col, iv_col, "TTE"):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in DataFrame.")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[k_col, iv_col, "TTE"]
    ).reset_index(drop=True)

    _empty = dict(n_left_put_otm=0, n_right_call_otm=0, n_atm=0,
                  n_total_after_filter=0, flag_unbalanced_smile=True)
    if df.empty:
        for c in ("abs_delta", "delta_min_tte", "smile_side"):
            df[c] = pd.NA
        df["keep_ssvi"] = False
        df.attrs.update(_empty)
        return df

    sigma = df[iv_col].to_numpy(float)
    T     = df["TTE"].to_numpy(float)
    k     = df[k_col].to_numpy(float)   # log(K/F) or log(K/S)

    # ── Forward Black-Scholes d1 ──────────────────────────────────────────────
    log_fk   = -k                                  # log(F/K) = −log(K/F)
    sqrt_T   = np.sqrt(np.maximum(T, 1e-8))
    sig_sqrt = sigma * sqrt_T
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.where(sig_sqrt > 1e-8,
                      (log_fk + 0.5 * sigma**2 * T) / sig_sqrt,
                      np.nan)

    call_delta = norm.cdf(d1)

    if "OptionType" in df.columns:
        is_call   = df["OptionType"].to_numpy() == 1
        abs_delta = np.where(is_call, call_delta, np.abs(call_delta - 1.0))
    else:
        abs_delta = np.where(k >= 0, call_delta, 1.0 - call_delta)

    # ── Smile-side classification ─────────────────────────────────────────────
    left_mask  = k < -atm_band
    right_mask = k >  atm_band
    atm_mask   = ~left_mask & ~right_mask
    smile_side = np.where(left_mask, "left_put_otm",
                 np.where(right_mask, "right_call_otm", "atm"))

    # ── Initial butterfly keep mask ───────────────────────────────────────────
    d_min = delta_min_tte(T, delta_short=delta_short, delta_long=delta_long, decay=decay)
    keep  = (np.isfinite(abs_delta) &
             (abs_delta >= d_min) &
             (abs_delta <= 1.0 - d_min)).copy()

    # ── Balance enforcement ───────────────────────────────────────────────────
    def _enforce_side(side_mask: np.ndarray, n_min: int) -> None:
        """Add back the closest-to-ATM discarded options from one side."""
        nonlocal keep
        n_have = int((keep & side_mask).sum())
        if n_have >= n_min:
            return
        n_add  = min(n_fallback, n_min - n_have)
        cand   = np.where(~keep & side_mask & np.isfinite(abs_delta))[0]
        if len(cand) == 0:
            return
        order = np.argsort(-abs_delta[cand])   # highest abs_delta first (closest to ATM)
        keep[cand[order[:n_add]]] = True

    _enforce_side(right_mask, min_right_points)
    _enforce_side(left_mask,  min_left_points)
    _enforce_side(atm_mask,   min_atm_points)

    # ── Final counts ──────────────────────────────────────────────────────────
    n_left  = int((keep & left_mask).sum())
    n_right = int((keep & right_mask).sum())
    n_atm_f = int((keep & atm_mask).sum())
    n_total = int(keep.sum())

    # ── Unbalanced-smile warnings ─────────────────────────────────────────────
    flag = False
    if n_right == 0:
        _warnings.warn(
            "Unbalanced smile: no OTM call (right side, k > 0) after delta filter.",
            UserWarning, stacklevel=2,
        )
        flag = True
    if n_left == 0:
        _warnings.warn(
            "Unbalanced smile: no OTM put (left side, k < 0) after delta filter.",
            UserWarning, stacklevel=2,
        )
        flag = True
    if n_total > 0:
        kept_k = k[keep]
        if kept_k.max() < atm_band:
            _warnings.warn(
                f"max(k)={kept_k.max():.4f} < {atm_band}: right wing has no coverage.",
                UserWarning, stacklevel=2,
            )
            flag = True
        if kept_k.min() > -atm_band:
            _warnings.warn(
                f"min(k)={kept_k.min():.4f} > {-atm_band}: left wing has no coverage.",
                UserWarning, stacklevel=2,
            )
            flag = True

    # ── Attach diagnostic columns ─────────────────────────────────────────────
    df["abs_delta"]     = abs_delta
    df["delta_min_tte"] = d_min
    df["keep_ssvi"]     = keep
    df["smile_side"]    = smile_side

    result = df[keep].reset_index(drop=True)
    result.attrs.update(dict(
        n_left_put_otm        = n_left,
        n_right_call_otm      = n_right,
        n_atm                 = n_atm_f,
        n_total_after_filter  = n_total,
        flag_unbalanced_smile = flag,
    ))
    return result


# =============================================================================
# 1. Maturity bucketing
# =============================================================================

def assign_maturity_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``maturity_bucket`` column derived from TTE.

    Buckets (TTE in years)
    ----------------------
    short  : 20d  ≤ TTE ≤  60d
    medium : 60d  < TTE ≤ 180d
    long   :        TTE > 180d
    other  :        TTE <  20d  (excluded from SSVI calibration)
    """
    def _label(tte: float) -> str:
        if tte < _BUCKET_LO["short"]:
            return "other"
        elif tte <= _BUCKET_HI["short"]:
            return "short"
        elif tte <= _BUCKET_HI["medium"]:
            return "medium"
        else:
            return "long"

    df = df.copy()
    df["maturity_bucket"] = df["TTE"].apply(_label)
    return df


# =============================================================================
# 2. SSVI formula
# =============================================================================

def ssvi_total_variance(
    k: np.ndarray,
    theta: float,
    rho: float,
    phi: float,
) -> np.ndarray:
    """SSVI total implied variance w(k; theta, rho, phi).

    w(k=0) = theta  (ATM total variance).
    The discriminant is always >= 1 - rho^2 > 0 for |rho| < 1.
    """
    k = np.asarray(k, dtype=float)
    disc = np.maximum((phi * k + rho) ** 2 + (1.0 - rho ** 2), 0.0)
    return (theta / 2.0) * (1.0 + rho * phi * k + np.sqrt(disc))


def ssvi_implied_vol(
    k: np.ndarray,
    T: np.ndarray,
    theta: float,
    rho: float,
    phi: float,
) -> np.ndarray:
    """Implied volatility from SSVI: sigma(k, T) = sqrt(w(k) / T).

    Returns np.nan where T <= 0 or w <= 0.
    """
    k = np.asarray(k, dtype=float)
    T = np.asarray(T, dtype=float)
    w = ssvi_total_variance(k, theta, rho, phi)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((w > 0) & (T > 0), np.sqrt(w / T), np.nan)


# =============================================================================
# 3. Data preparation
# =============================================================================

_REQUIRED_COLS = {"TTE", "implied_vol"}   # moneyness is auto-detected
_USEFUL_COLS   = [
    "Time Elapsed", "Moneyness", "forward_moneyness",
    "TTE", "implied_vol", "implied_vol_forward",
    "Liquidity Factor", "OPEN INTEREST", "OptionType",
]


def prepare_ssvi_bucket_slice(
    df: pd.DataFrame,
    time_elapsed,
    bucket: str,
    min_obs: int         = 40,
    max_liquidity: float = 0.10,
    min_iv: float        = 0.03,
    max_iv: float        = 2.00,
    min_tte: float       = 14 / 365,
    delta_short: float   = SSVI_DELTA_SHORT,
    delta_long:  float   = SSVI_DELTA_LONG,
    delta_decay: float   = SSVI_DELTA_DECAY,
    min_left_points:  int   = SSVI_MIN_LEFT_POINTS,
    min_right_points: int   = SSVI_MIN_RIGHT_POINTS,
    min_atm_points:   int   = SSVI_MIN_ATM_POINTS,
    atm_band:         float = SSVI_ATM_BAND,
    n_fallback:       int   = SSVI_FALLBACK_N,
) -> pd.DataFrame:
    """Extract and clean the slice for one (date, maturity bucket).

    Cleaning steps (in order)
    -------------------------
    1. Select rows: Time Elapsed == time_elapsed
    2. Select rows: TTE within bucket bounds
    3. Drop NaN / Inf in critical columns
    4. implied_vol in (min_iv, max_iv)
    5. Delta butterfly filter with balance enforcement
    6. TTE >= min_tte
    7. Liquidity Factor <= max_liquidity

    Raises
    ------
    ValueError  fewer than min_obs rows survive
    KeyError    required columns are absent
    """
    if bucket not in BUCKET_NAMES:
        raise ValueError(f"bucket must be one of {BUCKET_NAMES}, got {bucket!r}")

    present = [c for c in _USEFUL_COLS if c in df.columns]
    if not {"TTE", "implied_vol"}.issubset(present) and not {"TTE", "implied_vol_forward"}.issubset(present):
        raise KeyError("Required columns missing: need TTE + implied_vol (or implied_vol_forward).")

    sl = df.loc[df["Time Elapsed"] == time_elapsed, present].copy()

    # Bucket TTE range
    sl = sl[(sl["TTE"] >= _BUCKET_LO[bucket]) & (sl["TTE"] <= _BUCKET_HI[bucket])]

    # Clean NaN / Inf
    iv_col = "implied_vol_forward" if "implied_vol_forward" in sl.columns else "implied_vol"
    sl = sl.replace([np.inf, -np.inf], np.nan).dropna(subset=["TTE", iv_col])

    # IV range
    sl = sl[(sl[iv_col] >= min_iv) & (sl[iv_col] <= max_iv)]

    # Delta butterfly filter with balance enforcement
    n_before = len(sl)
    sl = filter_delta_butterfly_continuous(
        sl, delta_short=delta_short, delta_long=delta_long, decay=delta_decay,
        min_left_points=min_left_points, min_right_points=min_right_points,
        min_atm_points=min_atm_points, atm_band=atm_band, n_fallback=n_fallback,
    )
    n_after = len(sl)

    # Save balance diagnostics before further filtering clears attrs
    _balance_attrs = {k: sl.attrs.get(k, v) for k, v in {
        "n_left_put_otm": 0, "n_right_call_otm": 0, "n_atm": 0,
        "flag_unbalanced_smile": False,
    }.items()}

    # TTE floor (extra guard beyond bucket bounds)
    sl = sl[sl["TTE"] >= min_tte]

    # Liquidity
    if "Liquidity Factor" in sl.columns:
        sl = sl[sl["Liquidity Factor"].fillna(np.inf) <= max_liquidity]

    # Standardise: if forward columns were used, expose them under base names
    if "Moneyness" not in sl.columns and "forward_moneyness" in sl.columns:
        sl["Moneyness"] = sl["forward_moneyness"]
    if "implied_vol" not in sl.columns and "implied_vol_forward" in sl.columns:
        sl["implied_vol"] = sl["implied_vol_forward"]

    sl = sl.reset_index(drop=True)
    sl.attrs["n_before_delta"] = n_before
    sl.attrs["n_after_delta"]  = n_after
    sl.attrs.update(_balance_attrs)

    if len(sl) < min_obs:
        raise ValueError(
            f"Only {len(sl)} obs for ({time_elapsed!r}, {bucket!r}) after cleaning — "
            f"need >= {min_obs}."
        )
    return sl


# =============================================================================
# 4. Finance-consistent initial guess
# =============================================================================

def initial_guess_finance(
    slice_df: pd.DataFrame,
    w_obs: np.ndarray | None = None,
) -> np.ndarray:
    """Data-driven starting point for (theta, rho, phi).

    theta_0  Median total variance w = sigma^2 * T for ATM options
             (|Moneyness| < 0.05).  Falls back to full-slice median if no
             ATM options exist.

    rho_0    Fixed at -0.5 — typical SPX skew starting point.

    phi_0    Estimated from the observed slope of w vs k near the ATM:
             dw/dk |_{k=0} = theta * rho * phi
             => phi_0 = slope_w / (theta_0 * rho_0)
             Clipped to [0.1, 10] to avoid degenerate starts.

    Parameters
    ----------
    w_obs : pre-computed total variance array (len == len(slice_df)); computed
            here from implied_vol and TTE when not supplied.
    """
    sl = slice_df
    if w_obs is None:
        w_obs = sl["implied_vol"].to_numpy(float) ** 2 * sl["TTE"].to_numpy(float)
    w_obs = np.asarray(w_obs, dtype=float)

    # --- theta ---------------------------------------------------------------
    k_arr = sl["Moneyness"].to_numpy(float)
    atm      = np.abs(k_arr) < 0.05
    put_wing = k_arr < -0.05
    theta0   = float(np.median(w_obs[atm]) if atm.sum() >= 3 else np.median(w_obs))
    theta0   = max(theta0, 1e-4)

    # --- rho -----------------------------------------------------------------
    rho0 = -0.5

    # --- phi: from slope of w vs k near ATM ----------------------------------
    if put_wing.sum() >= 3 and atm.sum() >= 3:
        w_put   = float(np.median(w_obs[put_wing]))
        w_atm   = float(np.median(w_obs[atm]))
        k_put   = float(np.median(k_arr[put_wing]))   # negative
        slope_w = (w_put - w_atm) / k_put
        denom   = theta0 * rho0   # negative
        phi0 = float(slope_w / denom) if abs(denom) > 1e-8 and np.isfinite(slope_w) else 1.0
        phi0 = float(np.clip(phi0, 0.1, 10.0))
    else:
        phi0 = 1.0

    return np.array([theta0, rho0, phi0])


# =============================================================================
# 5. Objective function — total variance space
# =============================================================================

def ssvi_objective(
    params: np.ndarray,
    k: np.ndarray,
    T: np.ndarray,
    w_obs: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted residuals in total-variance space for scipy least_squares.

    Residual vector
    ---------------
    [ sqrt(w_i) * (w_SSVI_i - w_obs_i),  ...,  pen_phi,  pen_rho ]

    The last two scalar elements are soft penalties:
    - pen_phi = _PENALTY * max(phi - 10, 0)   — penalise extreme curvature
    - pen_rho = _PENALTY * max(|rho| - 0.95, 0) — penalise near-unit correlation
    """
    theta, rho, phi = params

    if theta <= 0 or phi <= 0 or abs(rho) >= 0.999:
        return np.full(len(k) + 2, 1e4)

    w_model   = ssvi_total_variance(k, theta, rho, phi)
    residuals = np.where(np.isfinite(w_model), w_model - w_obs, 1e3)

    if weights is not None:
        residuals = residuals * np.sqrt(np.maximum(weights, 0.0))

    pen_phi = _PENALTY * max(phi - 10.0, 0.0)
    pen_rho = _PENALTY * max(abs(rho) - 0.95, 0.0)

    return np.append(residuals, [pen_phi, pen_rho])


# =============================================================================
# 6. Calibration
# =============================================================================

def _build_weights(sl: pd.DataFrame) -> np.ndarray:
    """Normalised weights = log1p(OI) / (1 + LF)."""
    n = len(sl)
    lf = sl["Liquidity Factor"].fillna(0.0).clip(lower=0.0).to_numpy(float) \
         if "Liquidity Factor" in sl.columns else np.zeros(n)
    w_liq = 1.0 / (1.0 + lf)

    oi = sl["OPEN INTEREST"].fillna(0.0).clip(lower=0.0).to_numpy(float) \
         if "OPEN INTEREST" in sl.columns else np.ones(n)
    w_oi = np.log1p(oi)

    w = w_liq * w_oi
    mean_w = w.mean()
    return w / (mean_w + 1e-8) if mean_w > 1e-8 else np.ones(n)


def calibrate_ssvi_bucket(
    df: pd.DataFrame,
    time_elapsed,
    bucket: str,
    use_weights: bool    = True,
    min_obs: int         = 40,
    max_liquidity: float = 0.10,
    min_iv: float        = 0.03,
    max_iv: float        = 2.00,
    delta_short: float   = SSVI_DELTA_SHORT,
    delta_long:  float   = SSVI_DELTA_LONG,
    delta_decay: float   = SSVI_DELTA_DECAY,
    min_left_points:  int   = SSVI_MIN_LEFT_POINTS,
    min_right_points: int   = SSVI_MIN_RIGHT_POINTS,
    min_atm_points:   int   = SSVI_MIN_ATM_POINTS,
    atm_band:         float = SSVI_ATM_BAND,
    n_fallback:       int   = SSVI_FALLBACK_N,
) -> dict:
    """Calibrate SSVI for one (date, maturity bucket).

    Returns
    -------
    dict : time_elapsed, bucket, theta, rho, phi,
           success, cost, rmse_w, rmse_iv, n_obs, n_obs_before_delta,
           n_left_put_otm, n_right_call_otm, n_atm, flag_unbalanced_smile,
           message
    """
    base = dict(
        time_elapsed=time_elapsed, bucket=bucket,
        theta=np.nan, rho=np.nan, phi=np.nan,
        success=False, cost=np.nan,
        rmse_w=np.nan, rmse_iv=np.nan,
        n_obs=0, n_obs_before_delta=0,
        n_left_put_otm=0, n_right_call_otm=0, n_atm=0,
        flag_unbalanced_smile=False,
        message="",
    )

    # ── 1. Clean slice ────────────────────────────────────────────────────────
    try:
        sl = prepare_ssvi_bucket_slice(
            df, time_elapsed, bucket,
            min_obs=min_obs, max_liquidity=max_liquidity,
            min_iv=min_iv, max_iv=max_iv,
            delta_short=delta_short, delta_long=delta_long, delta_decay=delta_decay,
            min_left_points=min_left_points, min_right_points=min_right_points,
            min_atm_points=min_atm_points, atm_band=atm_band, n_fallback=n_fallback,
        )
    except (ValueError, KeyError) as exc:
        base["message"] = str(exc)
        return base

    base["n_obs_before_delta"]    = sl.attrs.get("n_before_delta", len(sl))
    base["n_left_put_otm"]        = sl.attrs.get("n_left_put_otm", 0)
    base["n_right_call_otm"]      = sl.attrs.get("n_right_call_otm", 0)
    base["n_atm"]                 = sl.attrs.get("n_atm", 0)
    base["flag_unbalanced_smile"] = sl.attrs.get("flag_unbalanced_smile", False)

    k     = sl["Moneyness"].to_numpy(float)
    T     = sl["TTE"].to_numpy(float)
    w_obs = sl["implied_vol"].to_numpy(float) ** 2 * T

    # ── 2. Weights ────────────────────────────────────────────────────────────
    weights = _build_weights(sl) if use_weights else None

    # ── 3. Optimise ───────────────────────────────────────────────────────────
    x0 = initial_guess_finance(sl, w_obs=w_obs)

    try:
        opt = least_squares(
            ssvi_objective,
            x0,
            args=(k, T, w_obs, weights),
            bounds=(_LO, _HI),
            method="trf",
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
            max_nfev=10_000,
        )
    except Exception as exc:
        base["message"] = f"Optimiser error: {exc}"
        return base

    theta_h, rho_h, phi_h = opt.x

    # ── 4. RMSE in both spaces ─────────────────────────────────────────────────
    w_fit   = ssvi_total_variance(k, theta_h, rho_h, phi_h)
    sig_fit = ssvi_implied_vol(k, T, theta_h, rho_h, phi_h)
    sig_obs = sl["implied_vol"].to_numpy(float)

    ok_w  = np.isfinite(w_fit)
    ok_iv = np.isfinite(sig_fit)
    rmse_w  = float(np.sqrt(np.mean((w_fit[ok_w]   - w_obs[ok_w])   ** 2))) if ok_w.any()  else np.nan
    rmse_iv = float(np.sqrt(np.mean((sig_fit[ok_iv] - sig_obs[ok_iv]) ** 2))) if ok_iv.any() else np.nan

    return dict(
        time_elapsed=time_elapsed, bucket=bucket,
        theta=float(theta_h), rho=float(rho_h), phi=float(phi_h),
        success=bool(opt.success), cost=float(opt.cost),
        rmse_w=rmse_w, rmse_iv=rmse_iv, n_obs=len(sl),
        n_obs_before_delta    = sl.attrs.get("n_before_delta", len(sl)),
        n_left_put_otm        = sl.attrs.get("n_left_put_otm", 0),
        n_right_call_otm      = sl.attrs.get("n_right_call_otm", 0),
        n_atm                 = sl.attrs.get("n_atm", 0),
        flag_unbalanced_smile = sl.attrs.get("flag_unbalanced_smile", False),
        message=opt.message,
    )


def calibrate_ssvi_day(
    df: pd.DataFrame,
    time_elapsed,
    **kwargs,
) -> dict[str, dict]:
    """Calibrate SSVI for all three buckets on one date.

    Returns
    -------
    dict  bucket_name -> result_dict
    """
    return {bkt: calibrate_ssvi_bucket(df, time_elapsed, bkt, **kwargs)
            for bkt in BUCKET_NAMES}


# =============================================================================
# 7. Predictions
# =============================================================================

def predict_ssvi_for_slice(
    slice_df: pd.DataFrame,
    result: dict,
) -> pd.DataFrame:
    """Attach SSVI predictions to a slice.

    Added columns
    -------------
    w_obs               : implied_vol^2 * TTE
    ssvi_total_variance : w(k; theta, rho, phi)
    ssvi_implied_vol    : sqrt(w / T)
    ssvi_error_iv       : ssvi_implied_vol - implied_vol
    ssvi_error_w        : ssvi_total_variance - w_obs
    """
    df = slice_df.copy()
    theta, rho, phi = result["theta"], result["rho"], result["phi"]

    k = df["Moneyness"].to_numpy(float)
    T = df["TTE"].to_numpy(float)

    df["w_obs"]               = df["implied_vol"] ** 2 * T
    df["ssvi_total_variance"] = ssvi_total_variance(k, theta, rho, phi)
    df["ssvi_implied_vol"]    = ssvi_implied_vol(k, T, theta, rho, phi)
    df["ssvi_error_iv"]       = df["ssvi_implied_vol"] - df["implied_vol"]
    df["ssvi_error_w"]        = df["ssvi_total_variance"] - df["w_obs"]
    return df


# =============================================================================
# 8. Sanity checks
# =============================================================================

def sanity_check_result(result: dict) -> list[str]:
    """Print a structured report for one calibrated bucket.

    Returns a list of warning/error strings (empty = all OK).
    """
    theta   = result.get("theta",   np.nan)
    rho     = result.get("rho",     np.nan)
    phi     = result.get("phi",     np.nan)
    rmse_w  = result.get("rmse_w",  np.nan)
    rmse_iv = result.get("rmse_iv", np.nan)
    n_obs   = result.get("n_obs",   0)
    bucket  = result.get("bucket",  "?")

    issues = []
    sep = "─" * 56
    print(sep)
    print(f"  Bucket : {bucket:>8}   |   n_obs = {n_obs}")
    print(sep)

    # theta
    atm_vol_1y = theta ** 0.5 if (np.isfinite(theta) and theta > 0) else np.nan
    ok_flag = "" if theta > 0 else "  ← ERROR"
    print(f"  theta  = {theta:.5f}   sqrt(theta) = {atm_vol_1y:.4f}{ok_flag}")
    if not (np.isfinite(theta) and theta > 0):
        issues.append("theta <= 0 or NaN — invalid")

    # rho
    flag = "" if rho < 0 else "  ← WARNING: expected rho < 0 for SPX"
    print(f"  rho    = {rho:.5f}{flag}")
    if np.isfinite(rho) and rho >= 0:
        issues.append("rho >= 0 — check data; SPX skew is typically negative")

    # phi
    flag = "  ← WARNING: > 10 is suspicious" if phi > 10 else ""
    print(f"  phi    = {phi:.5f}{flag}")
    if np.isfinite(phi) and phi > 10:
        issues.append(f"phi = {phi:.2f} > 10 — may indicate cleaning issue or mixed maturities")

    # RMSE
    ok_w  = "OK"  if (np.isfinite(rmse_w)  and rmse_w  < 0.005) else "LARGE"
    ok_iv = "OK"  if (np.isfinite(rmse_iv) and rmse_iv < 0.010) else "LARGE"
    print(f"  RMSE_w = {rmse_w:.6f}  [{ok_w}]   RMSE_iv = {rmse_iv:.6f}  [{ok_iv}]")
    if np.isfinite(rmse_iv) and rmse_iv > 0.02:
        issues.append(f"RMSE_iv = {rmse_iv:.4f} > 0.02 — poor fit")

    # convergence
    if not result.get("success", False):
        issues.append(f"Optimizer did not converge: {result.get('message','')}")
        print(f"  success = False  ({result.get('message','')})")
    else:
        print(f"  success = True")

    print(sep)
    if issues:
        for iss in issues:
            print(f"  ⚠  {iss}")
    else:
        print("  ✓  All checks passed")
    print()

    return issues


# =============================================================================
# 9. Visualisation
# =============================================================================

def plot_ssvi_fit_bucket(
    slice_df: pd.DataFrame,
    result: dict,
    title: str | None = None,
) -> None:
    """Two-panel plot: observed vs fitted IV (left), w-residuals (right).

    Points coloured by TTE; fitted lines drawn at the 10th, 50th, 90th
    percentile TTE within the bucket.
    """
    df = slice_df.copy()
    if "ssvi_implied_vol" not in df.columns:
        df = predict_ssvi_for_slice(df, result)

    theta   = result.get("theta",   np.nan)
    rho     = result.get("rho",     np.nan)
    phi     = result.get("phi",     np.nan)
    rmse_iv = result.get("rmse_iv", np.nan)
    rmse_w  = result.get("rmse_w",  np.nan)
    bucket  = result.get("bucket",  "")

    tte_vals  = df["TTE"].to_numpy(float)
    tte_min   = tte_vals.min()
    tte_range = max(tte_vals.max() - tte_min, 1e-8)
    tte_norm  = (tte_vals - tte_min) / tte_range
    cmap      = plt.cm.viridis

    k_grid   = np.linspace(df["Moneyness"].min(), df["Moneyness"].max(), 300)
    tte_pcts = np.percentile(tte_vals, [10, 50, 90])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        title or (
            f"SSVI fit — {bucket} bucket\n"
            f"θ={theta:.4f}  ρ={rho:.3f}  φ={phi:.3f}  "
            f"RMSE_iv={rmse_iv:.4f}  RMSE_w={rmse_w:.5f}"
        ),
        fontsize=11,
    )

    # Observed scatter + fitted lines
    sc = ax1.scatter(df["Moneyness"], df["implied_vol"],
                     c=tte_norm, cmap=cmap, s=10, alpha=0.5, zorder=2, label="Observed")
    plt.colorbar(sc, ax=ax1, label="TTE (normalised)")
    for tte_v in tte_pcts:
        col     = cmap((tte_v - tte_min) / tte_range)
        iv_line = ssvi_implied_vol(k_grid, np.full_like(k_grid, tte_v), theta, rho, phi)
        ax1.plot(k_grid, iv_line, color=col, linewidth=2.0,
                 label=f"T={tte_v * 365:.0f}d")
    ax1.set_xlabel("Moneyness = log(K/S)")
    ax1.set_ylabel("Implied volatility")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(alpha=0.3)

    # Residuals in w-space
    ax2.scatter(df["Moneyness"], df["ssvi_error_w"],
                c=tte_norm, cmap=cmap, s=10, alpha=0.5)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Moneyness = log(K/S)")
    ax2.set_ylabel("w_SSVI − w_obs")
    ax2.set_title(f"Total-variance residuals   RMSE_w = {rmse_w:.5f}")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_ssvi_day(
    day_results: dict[str, dict],
    slice_dfs: dict[str, pd.DataFrame],
) -> None:
    """Side-by-side SSVI fit for all successfully calibrated buckets."""
    ok_buckets = [b for b in BUCKET_NAMES if day_results.get(b, {}).get("success")]
    if not ok_buckets:
        print("No successfully calibrated bucket for this date.")
        return

    n   = len(ok_buckets)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    te = day_results[ok_buckets[0]]["time_elapsed"]
    fig.suptitle(f"SSVI calibration — {te}", fontsize=12)

    cmap = plt.cm.viridis

    for ax, bkt in zip(axes, ok_buckets):
        res = day_results[bkt]
        df  = slice_dfs[bkt]

        tte_vals  = df["TTE"].to_numpy(float)
        tte_min   = tte_vals.min()
        tte_range = max(tte_vals.max() - tte_min, 1e-8)
        tte_norm  = (tte_vals - tte_min) / tte_range
        k_grid    = np.linspace(df["Moneyness"].min(), df["Moneyness"].max(), 300)
        tte_pcts  = np.percentile(tte_vals, [10, 50, 90])

        ax.scatter(df["Moneyness"], df["implied_vol"],
                   c=tte_norm, cmap=cmap, s=8, alpha=0.45, zorder=2)
        for tte_v in tte_pcts:
            col     = cmap((tte_v - tte_min) / tte_range)
            iv_line = ssvi_implied_vol(k_grid, np.full_like(k_grid, tte_v),
                                       res["theta"], res["rho"], res["phi"])
            ax.plot(k_grid, iv_line, color=col, linewidth=2.0, alpha=0.9)

        ax.set_title(
            f"{bkt}\n"
            f"θ={res['theta']:.4f}  ρ={res['rho']:.3f}  φ={res['phi']:.3f}\n"
            f"RMSE_iv={res['rmse_iv']:.4f}  n={res['n_obs']}",
            fontsize=9,
        )
        ax.set_xlabel("Moneyness")
        ax.set_ylabel("Implied vol")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
