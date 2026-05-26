"""HAR-RV feature construction and multi-step target generation."""

import numpy as np
import pandas as pd


def har_features(rv_daily: pd.Series) -> pd.DataFrame:
    """
    Build HAR-RV components from a daily realized volatility series.

    Corsi (2009) heterogeneous agent decomposition:
        RV_d = rv_daily  (daily)
        RV_w = 5-day  backward mean (weekly component)
        RV_m = 22-day backward mean (monthly component)

    All components are annualized, consistent with rv_daily being annualized.

    Parameters
    ----------
    rv_daily : Series of annualized daily RV values

    Returns
    -------
    DataFrame with columns: rv_d, rv_w, rv_m, log_rv_d, log_rv_w, log_rv_m
    """
    rv_d = rv_daily.copy()
    rv_w = rv_d.rolling(5,  min_periods=5).mean()
    rv_m = rv_d.rolling(22, min_periods=22).mean()

    log_rv_d = np.log(rv_d.clip(lower=1e-12))
    log_rv_w = np.log(rv_w.clip(lower=1e-12))
    log_rv_m = np.log(rv_m.clip(lower=1e-12))

    return pd.DataFrame({
        'rv_d':     rv_d,
        'rv_w':     rv_w,
        'rv_m':     rv_m,
        'log_rv_d': log_rv_d,
        'log_rv_w': log_rv_w,
        'log_rv_m': log_rv_m,
    }, index=rv_daily.index)


def make_log_rv_target(rv_daily: pd.Series, h: int) -> pd.Series:
    """
    Multi-step forward log-RV target for horizon h.

    y_t^(h) = (1/h) * sum_{i=1}^{h} log(RV_{t+i})
            = rolling(h).mean().shift(-h) applied to log_rv_daily

    The rolling mean is computed first (backward), then shifted backward in
    calendar time so that y_t^(h) aligns with features at time t.
    No look-ahead: features at t use only data up to t; target y_t^(h)
    uses returns at t+1..t+h.

    Parameters
    ----------
    rv_daily : annualized daily RV series
    h        : forecast horizon in trading days

    Returns
    -------
    Series of log-RV targets, NaN for last h observations.
    """
    log_rv = np.log(rv_daily.clip(lower=1e-12))
    return log_rv.rolling(h).mean().shift(-h)


def compute_rv_from_returns(log_returns: pd.Series) -> pd.Series:
    """
    Annualized daily realized volatility proxy from log-returns.
    RV_t = sqrt(252) * |r_t|   (assumes iid Gaussian daily returns)
    Equivalently: annualized variance proxy = 252 * r_t^2.
    """
    return np.sqrt(252.0 * log_returns ** 2)


def smooth_stress_indicator(vix_series: pd.Series,
                             window: int = 252,
                             min_periods: int = 60,
                             threshold: float = 0.70,
                             slope: float = 10.0) -> pd.Series:
    """
    Smooth regime stress indicator based on rolling VIX percentile rank.

    lambda_t = sigmoid(slope * (VIX_pct_t - threshold))
             = 1 / (1 + exp(-slope * (VIX_pct_t - threshold)))

    lambda_t in [0, 1]: 1 = high-stress regime, 0 = calm regime.
    Smooth transition avoids discontinuity of hard threshold models (Hansen 1999).

    Parameters
    ----------
    vix_series  : Series of daily VIX values (decimal or percentage, consistent)
    window      : rolling window for percentile rank
    min_periods : minimum observations to compute percentile
    threshold   : percentile rank threshold (default: 70th percentile)
    slope       : sigmoid steepness (default: 10)

    Returns
    -------
    Series of lambda_t values in [0, 1].
    """
    vix_pct = vix_series.rolling(window, min_periods=min_periods).apply(
        lambda x: float((x <= x.iloc[-1]).mean()), raw=False
    ).fillna(0.5)
    return 1.0 / (1.0 + np.exp(-slope * (vix_pct - threshold)))
