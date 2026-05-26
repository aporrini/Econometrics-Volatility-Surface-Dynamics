"""Proper scoring rules for interval and quantile forecasts."""

import numpy as np


def pinball_loss(y_true, q_hat, tau: float) -> float:
    """
    Pinball (quantile) loss — proper scoring rule.

    rho_tau(y - q) = tau * max(y-q, 0) + (1-tau) * max(q-y, 0)

    Parameters
    ----------
    y_true : array of realized values
    q_hat  : array of quantile forecasts
    tau    : quantile level (e.g., 0.10 for lower 10th percentile)

    Returns
    -------
    Mean pinball loss (scalar, lower is better).

    Reference
    ---------
    Koenker & Bassett (1978), Econometrica 46(1), 33–50.
    """
    err = np.asarray(y_true) - np.asarray(q_hat)
    return float(np.mean(np.where(err >= 0, tau * err, (tau - 1) * err)))


def winkler_score(y_true, lower, upper, alpha: float = 0.20) -> float:
    """
    Winkler (1972) interval score: penalizes width and coverage violations.

    Score = (upper - lower) + (2/alpha) * [violation magnitude]

    Parameters
    ----------
    y_true : array of realized values
    lower  : array of lower interval bounds (e.g., 10th percentile)
    upper  : array of upper interval bounds (e.g., 90th percentile)
    alpha  : nominal miscoverage (0.20 for 80% PI)

    Returns
    -------
    Mean Winkler score (scalar, lower is better).

    Reference
    ---------
    Winkler, R.L. (1972), JASA 67(337), 675–679.
    """
    y     = np.asarray(y_true)
    lo    = np.asarray(lower)
    hi    = np.asarray(upper)
    width = hi - lo
    below = np.maximum(lo - y, 0)
    above = np.maximum(y - hi, 0)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


def coverage(y_true, lower, upper) -> float:
    """Empirical coverage of prediction intervals."""
    y  = np.asarray(y_true)
    lo = np.asarray(lower)
    hi = np.asarray(upper)
    return float(np.mean((y >= lo) & (y <= hi)))
