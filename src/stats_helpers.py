"""Statistical helpers: OLS prediction, DM-HLN test, R²_OOS, ARMA selection."""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS


# ── OLS prediction ────────────────────────────────────────────────────────────

def fit_ols_predict(X_tr, y_tr, X_te):
    """
    Fit OLS on (X_tr, y_tr) and predict on X_te.
    A constant is added automatically.

    Returns
    -------
    model    : fitted statsmodels RegressionResultsWrapper
    y_hat_te : ndarray of predictions on test set
    """
    Xc_tr = sm.add_constant(X_tr, has_constant='add')
    model = OLS(y_tr, Xc_tr).fit()
    Xc_te = sm.add_constant(X_te, has_constant='add')
    Xc_te = Xc_te.reindex(columns=Xc_tr.columns, fill_value=0)
    return model, model.predict(Xc_te)


# ── Forecast accuracy metrics ─────────────────────────────────────────────────

def r2_oos(y_true, y_pred, y_naive):
    """
    Out-of-sample R² relative to the Naive (random-walk) benchmark.
        R²_OOS = 1 - MSE_model / MSE_naive
    Positive values indicate the model beats the naive benchmark.
    """
    mse_m = np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)
    mse_n = np.mean((np.asarray(y_true) - np.asarray(y_naive)) ** 2)
    return float(1 - mse_m / mse_n)


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error (%).
    Uses eps to avoid division by zero.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred) / np.abs(y_true).clip(eps)) * 100)


# ── DM-HLN test ───────────────────────────────────────────────────────────────

def dm_hln(e_model, e_bench, h: int = 1):
    """
    Diebold-Mariano test with Harvey-Leybourne-Newbold small-sample correction.

    d_t = e_bench_t² − e_model_t²  (positive → model beats bench)
    HAC variance via Bartlett kernel with h−1 lags.
    HLN correction: kappa = (T + 1 − 2h + h(h−1)/T) / T

    Parameters
    ----------
    e_model, e_bench : arrays of forecast errors
    h                : forecast horizon (used for HAC bandwidth and HLN)

    Returns
    -------
    dm_stat : float  (positive → model better than bench)
    p_value : float  (two-sided)

    References
    ----------
    Diebold & Mariano (1995), JBES 13(3), 253–263.
    Harvey, Leybourne & Newbold (1997), IJF 13(2), 281–291.
    """
    d = np.asarray(e_bench) ** 2 - np.asarray(e_model) ** 2
    T = len(d)
    d_bar  = d.mean()
    gamma0 = np.var(d, ddof=1)
    hac    = gamma0
    for lag in range(1, h):
        if lag < T:
            gl   = np.cov(d[lag:], d[:-lag])[0, 1]
            hac += 2 * (1 - lag / h) * gl    # Bartlett kernel
    kappa  = (T + 1 - 2 * h + h * (h - 1) / T) / T
    dm_val = d_bar / np.sqrt(max(hac * kappa / T, 1e-20))
    p_val  = 2 * stats.norm.cdf(-abs(dm_val))
    return float(dm_val), float(p_val)


# ── ARMA selection ────────────────────────────────────────────────────────────

def select_arma_order(series, max_p: int = 5, max_q: int = 5,
                       ic: str = 'aic') -> tuple[int, int]:
    """
    Grid-search ARMA(p, q) by AIC or BIC over p ∈ [0, max_p], q ∈ [0, max_q].

    Parameters
    ----------
    series : array-like or Series (stationary)
    max_p  : maximum AR order
    max_q  : maximum MA order
    ic     : 'aic' or 'bic'

    Returns
    -------
    (p_best, q_best) tuple
    """
    from statsmodels.tsa.arima.model import ARIMA
    best_ic, best_pq = np.inf, (1, 0)
    y = np.asarray(series.dropna() if hasattr(series, 'dropna') else series)
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            if p + q == 0:
                continue
            try:
                res = ARIMA(y, order=(p, 0, q)).fit()
                val = res.aic if ic == 'aic' else res.bic
                if val < best_ic:
                    best_ic, best_pq = val, (p, q)
            except Exception:
                pass
    return best_pq


def rolling_arma_oos(series: pd.Series, p: int, q: int,
                      n_train: int) -> pd.Series:
    """
    Expanding-window OOS ARMA(p,q) one-step-ahead forecasts.

    Parameters
    ----------
    series  : full time series (train + test)
    p, q    : ARMA order
    n_train : initial training window size

    Returns
    -------
    Series of OOS forecasts, NaN for first n_train observations.
    """
    from statsmodels.tsa.arima.model import ARIMA
    forecasts = pd.Series(np.nan, index=series.index)
    for i in range(n_train, len(series)):
        y_tr = series.iloc[:i].dropna()
        try:
            res  = ARIMA(y_tr, order=(p, 0, q)).fit()
            forecasts.iloc[i] = float(res.forecast(1).iloc[0])
        except Exception:
            forecasts.iloc[i] = y_tr.iloc[-1]   # fallback: last value
    return forecasts


def rolling_armax_oos(endog: pd.Series, exog: pd.DataFrame,
                       p: int, q: int, n_train: int) -> pd.Series:
    """
    Expanding-window OOS ARMAX(p,q) one-step-ahead forecasts.
    Exogenous variables are pre-lagged (shift(1)) before calling this function.

    Parameters
    ----------
    endog   : endogenous series
    exog    : exogenous regressors (same index as endog, already lagged)
    p, q    : ARMA order
    n_train : initial training window size

    Returns
    -------
    Series of OOS forecasts.
    """
    from statsmodels.tsa.arima.model import ARIMA
    forecasts = pd.Series(np.nan, index=endog.index)
    for i in range(n_train, len(endog)):
        y_tr  = endog.iloc[:i].dropna()
        X_tr  = exog.iloc[:i].loc[y_tr.index]
        # one-step forecast uses exog at t (already lagged, so no look-ahead)
        X_fc  = exog.iloc[[i]]
        try:
            res = ARIMA(y_tr, order=(p, 0, q), exog=X_tr).fit()
            forecasts.iloc[i] = float(
                res.forecast(1, exog=X_fc).iloc[0]
            )
        except Exception:
            forecasts.iloc[i] = y_tr.iloc[-1]
    return forecasts


# ── Chow structural break test ────────────────────────────────────────────────

def chow_test(y: np.ndarray, X: np.ndarray, break_idx: int) -> tuple[float, float]:
    """
    Chow (1960) test for a structural break at break_idx.

    F = [(SSR_pool - SSR_1 - SSR_2) / k] / [(SSR_1 + SSR_2) / (n - 2k)]

    Parameters
    ----------
    y         : dependent variable
    X         : regressor matrix (WITHOUT constant)
    break_idx : integer index of the break point

    Returns
    -------
    (F_stat, p_value)
    """
    def ssr(y_, X_):
        Xc = np.column_stack([np.ones(len(y_)), X_])
        b  = np.linalg.lstsq(Xc, y_, rcond=None)[0]
        return float(np.sum((y_ - Xc @ b) ** 2))

    k = X.shape[1] + 1   # number of parameters (including constant)
    n = len(y)
    SSR_pool = ssr(y, X)
    SSR_1    = ssr(y[:break_idx], X[:break_idx])
    SSR_2    = ssr(y[break_idx:], X[break_idx:])
    F_stat   = ((SSR_pool - SSR_1 - SSR_2) / k) / ((SSR_1 + SSR_2) / (n - 2 * k))
    p_val    = 1 - stats.f.cdf(F_stat, dfn=k, dfd=n - 2 * k)
    return float(F_stat), float(p_val)
