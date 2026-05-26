"""SSVI-derived features for the econometric analysis notebooks."""

import numpy as np
import pandas as pd


def ssvi_atm_iv(alpha: pd.Series, beta: pd.Series, T: float = 0.25) -> pd.Series:
    """
    Annualized ATM implied volatility at reference maturity T.

    From the SSVI formula at k=0:
        omega(0, T) = theta_T = exp(alpha) * T^beta
        sigma_ATM   = sqrt(theta_T / T) = exp(alpha/2) * T^((beta-1)/2)

    Parameters
    ----------
    alpha, beta : Series with DatetimeIndex
    T           : reference maturity in years (default 0.25 = 3 months)

    Returns
    -------
    Series of annualized ATM IV, same index as alpha.
    """
    theta = np.exp(alpha) * (T ** beta)
    return np.sqrt(theta / T)


def ssvi_omega(k, alpha: float, beta: float, rho: float, eta: float, gamma: float,
               T: float = 0.25) -> float:
    """
    SSVI total implied variance at log-moneyness k and maturity T.

    Gatheral & Jacquier (2014) parametrization:
        omega(k, T) = (theta/2) * {1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1 - rho^2)}
        theta = exp(alpha) * T^beta
        phi   = eta * theta^(-gamma) / (1 + eta * theta^(1-gamma))
    """
    theta = np.exp(alpha) * (T ** beta)
    phi   = eta * theta ** (-gamma) / (1 + eta * theta ** (1 - gamma))
    return (theta / 2) * (
        1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho ** 2)
    )


def ssvi_derived_features(ssvi_df: pd.DataFrame, T_ref: float = 0.25) -> pd.DataFrame:
    """
    Compute the full set of SSVI-derived features used in Notebooks B and C.

    Parameters
    ----------
    ssvi_df : DataFrame with columns ['alpha','beta','rho','eta','gamma']
    T_ref   : reference maturity for ATM IV (years)

    Returns
    -------
    DataFrame with additional columns: atm_iv, log_atm_iv, abs_rho, skew_stress,
    eta_gamma, d_alpha, d_beta, d_rho, d_eta, d_gamma
    """
    df = ssvi_df.copy()
    params = ['alpha', 'beta', 'rho', 'eta', 'gamma']
    for p in params:
        if p not in df.columns:
            raise ValueError(f"Missing column: '{p}'")

    df['atm_iv']     = ssvi_atm_iv(df['alpha'], df['beta'], T=T_ref)
    df['log_atm_iv'] = np.log(df['atm_iv'].clip(lower=1e-12))
    df['abs_rho']    = df['rho'].abs()
    df['skew_stress'] = df['abs_rho'] * df['eta']
    df['eta_gamma']  = df['eta'] * df['gamma']

    for p in params:
        df[f'd_{p}'] = df[p].diff()

    return df
