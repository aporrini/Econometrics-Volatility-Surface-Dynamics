"""
ssvi_projection.py
------------------
Post-hoc projection of SSVI parameter vectors onto the no-arbitrage feasible set.

SSVI parameterisation (Gatheral & Jacquier 2014):
    w(k, T) = theta(T)/2 * (1 + rho*phi*k + sqrt((phi*k+rho)^2 + 1-rho^2))
    theta(T) = exp(alpha) * T^beta          (ATM total variance)
    phi(T)   = eta / theta(T)^gamma         (smile slope)

No-arbitrage sufficient conditions (butterfly):
    cond1(T) = theta(T) * phi(T)   * (1+|rho|) < 4   for all T
    cond2(T) = theta(T) * phi(T)^2 * (1+|rho|) < 4   for all T

Projection strategy (analytical, O(n * |T_grid|)):
    1. Clip rho   to (-1, 1) — hard bounds
    2. Clip beta  to [0, inf)
    3. Clip gamma to (0, 1]
    4. Clip eta   to (0, inf)
    5. For cond1: eta_max = min_T  4*(1-eps) / (theta^(1-gamma) * (1+|rho|))
    6. For cond2: eta_max = min_T  2*(1-eps) / sqrt(theta^(1-2*gamma) * (1+|rho|))
    7. eta_proj  = min(eta_clipped, eta_max_cond1, eta_max_cond2)

NOTE on alpha:
    alpha is unconstrained — exp(alpha) > 0 always.
    In our SPX calibrations alpha ~ -4 (negative). Do NOT clip alpha.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# ── Default T grid ─────────────────────────────────────────────────────────────
# Covers typical SPX option maturities: 2 weeks to 2 years
_T_DEFAULT = np.concatenate([
    np.linspace(14/365, 0.5,  40),
    np.linspace(0.5,    2.0,  60),
])

# ── Feasible bounds ────────────────────────────────────────────────────────────
_RHO_LO,  _RHO_HI  = -0.9999, 0.9999
_ETA_MIN            = 1e-6
_BETA_MIN           = 0.0
_GAMMA_LO, _GAMMA_HI = 1e-6, 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Core analytical projection (single vector)
# ─────────────────────────────────────────────────────────────────────────────

def project_one(
    alpha: float,
    beta:  float,
    rho:   float,
    eta:   float,
    gamma: float,
    T_grid: np.ndarray | None = None,
    epsilon: float = 0.01,
) -> tuple[float, float, float, float, float, dict]:
    """
    Project a single (alpha, beta, rho, eta, gamma) onto the feasible set.

    Returns
    -------
    (alpha_p, beta_p, rho_p, eta_p, gamma_p, info_dict)

    info_dict keys
    --------------
    clipped_rho, clipped_beta, clipped_gamma, clipped_eta_bounds,
    clipped_eta_cond1, clipped_eta_cond2,
    max_cond1_before, max_cond2_before,
    max_cond1_after,  max_cond2_after,
    any_change
    """
    if T_grid is None:
        T_grid = _T_DEFAULT

    info: dict = {}

    # ── Step 1: bound clips ───────────────────────────────────────────────────
    rho_p   = float(np.clip(rho,   _RHO_LO,   _RHO_HI))
    beta_p  = float(max(beta,  _BETA_MIN))
    gamma_p = float(np.clip(gamma, _GAMMA_LO, _GAMMA_HI))
    eta_p   = float(max(eta,   _ETA_MIN))
    alpha_p = float(alpha)  # unconstrained

    info["clipped_rho"]         = rho_p   != float(rho)
    info["clipped_beta"]        = beta_p  != float(beta)
    info["clipped_gamma"]       = gamma_p != float(gamma)
    info["clipped_eta_bounds"]  = eta_p   != float(eta)

    # ── Shared quantities ─────────────────────────────────────────────────────
    abs_rho = abs(rho_p)
    theta   = np.exp(alpha_p) * T_grid ** beta_p       # shape (n_T,)
    phi     = eta_p / theta ** gamma_p                  # shape (n_T,)

    # Conditions BEFORE butterfly clipping
    info["max_cond1_before"] = float(np.max(theta * phi     * (1 + abs_rho)))
    info["max_cond2_before"] = float(np.max(theta * phi**2  * (1 + abs_rho)))

    # ── Step 2: cond1 analytical bound on eta ────────────────────────────────
    # cond1: theta * phi * (1+|rho|) = eta * theta^(1-gamma) * (1+|rho|) < 4
    # => eta < 4 / (theta^(1-gamma) * (1+|rho|))  for every T
    # => eta_max_c1 = min_T( 4*(1-eps) / (theta^(1-gamma)*(1+|rho|)) )
    theta_pow1 = theta ** (1 - gamma_p)                 # theta^(1-gamma)
    denom1     = theta_pow1 * (1 + abs_rho)
    eta_max_c1 = float(np.min(4 * (1 - epsilon) / np.maximum(denom1, 1e-30)))

    info["clipped_eta_cond1"] = eta_p > eta_max_c1
    eta_p = min(eta_p, eta_max_c1)

    # ── Step 3: cond2 analytical bound on eta ────────────────────────────────
    # cond2: theta * phi^2 * (1+|rho|) = eta^2 * theta^(1-2*gamma) * (1+|rho|) < 4
    # => eta < 2 / sqrt(theta^(1-2*gamma) * (1+|rho|))  for every T
    # => eta_max_c2 = min_T( 2*(1-eps) / sqrt(theta^(1-2*gamma)*(1+|rho|)) )
    theta_pow2 = theta ** (1 - 2 * gamma_p)             # theta^(1-2*gamma)
    denom2     = np.maximum(theta_pow2 * (1 + abs_rho), 1e-30)
    eta_max_c2 = float(np.min(2 * (1 - epsilon) / np.sqrt(denom2)))

    info["clipped_eta_cond2"] = eta_p > eta_max_c2
    eta_p = min(eta_p, eta_max_c2)

    # ── Conditions AFTER projection ───────────────────────────────────────────
    phi_p = eta_p / theta ** gamma_p
    info["max_cond1_after"] = float(np.max(theta * phi_p     * (1 + abs_rho)))
    info["max_cond2_after"] = float(np.max(theta * phi_p**2  * (1 + abs_rho)))
    info["any_change"] = any([
        info["clipped_rho"], info["clipped_beta"], info["clipped_gamma"],
        info["clipped_eta_bounds"], info["clipped_eta_cond1"], info["clipped_eta_cond2"],
    ])

    return alpha_p, beta_p, rho_p, eta_p, gamma_p, info


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised projection (DataFrame in, DataFrame out)
# ─────────────────────────────────────────────────────────────────────────────

def project_params(
    df: pd.DataFrame,
    T_grid: np.ndarray | None = None,
    epsilon: float = 0.01,
    param_cols: tuple[str, ...] = ("alpha", "beta", "rho", "eta", "gamma"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Vectorised post-hoc projection of SSVI parameters.

    Parameters
    ----------
    df        : DataFrame containing columns in param_cols
    T_grid    : maturity grid (years). Default: 100 points from 14d to 2y.
    epsilon   : safety margin — projected values satisfy cond < 4*(1-epsilon)
    param_cols: column names for (alpha, beta, rho, eta, gamma)

    Returns
    -------
    projected : DataFrame — same shape as df, with param_cols replaced by
                projected values (non-param columns are passed through unchanged)
    flags     : DataFrame — boolean columns per violation type + summary cols
    """
    if T_grid is None:
        T_grid = _T_DEFAULT

    alpha_c, beta_c, rho_c, eta_c, gamma_c = param_cols

    a = df[alpha_c].to_numpy(dtype=float)
    b = df[beta_c].to_numpy(dtype=float)
    r = df[rho_c].to_numpy(dtype=float)
    e = df[eta_c].to_numpy(dtype=float)
    g = df[gamma_c].to_numpy(dtype=float)
    n = len(a)

    # ── Flags dataframe ───────────────────────────────────────────────────────
    flags = pd.DataFrame(index=df.index)

    # ── Step 1: bound clips ───────────────────────────────────────────────────
    r_p = np.clip(r, _RHO_LO,   _RHO_HI)
    b_p = np.maximum(b, _BETA_MIN)
    g_p = np.clip(g, _GAMMA_LO, _GAMMA_HI)
    e_p = np.maximum(e, _ETA_MIN)
    a_p = a.copy()

    flags["clip_rho"]         = r_p != r
    flags["clip_beta"]        = b_p != b
    flags["clip_gamma"]       = g_p != g
    flags["clip_eta_bounds"]  = e_p != e

    # ── Shared arrays: theta on grid (n, n_T) ────────────────────────────────
    # theta[i, j] = exp(a_p[i]) * T_grid[j]^b_p[i]
    log_theta = (a_p[:, None]
                 + b_p[:, None] * np.log(T_grid[None, :]))  # (n, n_T)
    theta     = np.exp(log_theta)                            # (n, n_T)
    abs_rho   = np.abs(r_p)[:, None]                        # (n, 1)

    # max_cond1 / max_cond2 BEFORE butterfly clipping  (using current e_p)
    phi_before = e_p[:, None] / theta ** g_p[:, None]
    flags["max_cond1_before"] = np.max(theta * phi_before     * (1 + abs_rho), axis=1)
    flags["max_cond2_before"] = np.max(theta * phi_before**2  * (1 + abs_rho), axis=1)
    flags["butterfly_viol_before"] = (
        (flags["max_cond1_before"] >= 4.0) | (flags["max_cond2_before"] >= 4.0)
    )

    # ── Step 2: cond1 analytical bound ───────────────────────────────────────
    # eta_max_c1[i] = min_T( 4*(1-eps) / (theta[i,j]^(1-g_p[i]) * (1+|r_p[i]|)) )
    theta_pow1  = theta ** (1 - g_p[:, None])               # (n, n_T)
    denom1      = np.maximum(theta_pow1 * (1 + abs_rho), 1e-30)
    eta_max_c1  = np.min(4 * (1 - epsilon) / denom1, axis=1)  # (n,)

    flags["clip_eta_cond1"] = e_p > eta_max_c1
    e_p = np.minimum(e_p, eta_max_c1)

    # ── Step 3: cond2 analytical bound ───────────────────────────────────────
    # eta_max_c2[i] = min_T( 2*(1-eps) / sqrt(theta[i,j]^(1-2*g_p[i]) * (1+|r_p[i]|)) )
    theta_pow2  = theta ** (1 - 2 * g_p[:, None])           # (n, n_T)
    denom2      = np.maximum(theta_pow2 * (1 + abs_rho), 1e-30)
    eta_max_c2  = np.min(2 * (1 - epsilon) / np.sqrt(denom2), axis=1)  # (n,)

    flags["clip_eta_cond2"] = e_p > eta_max_c2
    e_p = np.minimum(e_p, eta_max_c2)

    # ── max_cond1 / max_cond2 AFTER projection ────────────────────────────────
    phi_after = e_p[:, None] / theta ** g_p[:, None]
    flags["max_cond1_after"] = np.max(theta * phi_after     * (1 + abs_rho), axis=1)
    flags["max_cond2_after"] = np.max(theta * phi_after**2  * (1 + abs_rho), axis=1)
    flags["butterfly_viol_after"] = (
        (flags["max_cond1_after"] >= 4.0) | (flags["max_cond2_after"] >= 4.0)
    )

    # Summary columns
    flags["any_change"] = (
        flags["clip_rho"]  | flags["clip_beta"]  | flags["clip_gamma"] |
        flags["clip_eta_bounds"] | flags["clip_eta_cond1"] | flags["clip_eta_cond2"]
    )

    # ── Build projected DataFrame ─────────────────────────────────────────────
    projected = df.copy()
    projected[alpha_c] = a_p
    projected[beta_c]  = b_p
    projected[rho_c]   = r_p
    projected[eta_c]   = e_p
    projected[gamma_c] = g_p

    return projected, flags


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def violation_report(flags: pd.DataFrame, label: str = "") -> None:
    """Print a concise violation/projection report from the flags DataFrame."""
    n = len(flags)
    sep = "-" * 55
    print(sep)  # noqa: ascii-safe
    if label:
        print(f"  Projection Report — {label}")
    print(sep)  # noqa: ascii-safe
    print(f"  Total observations     : {n}")
    print()
    print("  Violations BEFORE projection:")
    print(f"    Butterfly (cond1 or cond2 >= 4) : "
          f"{flags['butterfly_viol_before'].sum():>3d} / {n}  "
          f"({100*flags['butterfly_viol_before'].mean():.1f}%)")
    print(f"    max cond1 mean/max  : "
          f"{flags['max_cond1_before'].mean():.3f} / {flags['max_cond1_before'].max():.3f}")
    print(f"    max cond2 mean/max  : "
          f"{flags['max_cond2_before'].mean():.3f} / {flags['max_cond2_before'].max():.3f}")
    print()
    print("  Clipping breakdown:")
    for col in ["clip_rho","clip_beta","clip_gamma","clip_eta_bounds",
                "clip_eta_cond1","clip_eta_cond2"]:
        if col in flags.columns:
            cnt = flags[col].sum()
            if cnt > 0:
                print(f"    {col:<25s}: {cnt:>3d}  ({100*cnt/n:.1f}%)")
    print(f"    {'any_change':<25s}: {flags['any_change'].sum():>3d}  "
          f"({100*flags['any_change'].mean():.1f}%)")
    print()
    print("  Violations AFTER projection:")
    remaining = flags["butterfly_viol_after"].sum()
    print(f"    Butterfly remaining        : {remaining:>3d} / {n}  "
          f"({'NONE — all fixed' if remaining == 0 else 'WARNING: still violated'})")
    print(f"    max cond1 mean/max  : "
          f"{flags['max_cond1_after'].mean():.3f} / {flags['max_cond1_after'].max():.3f}")
    print(f"    max cond2 mean/max  : "
          f"{flags['max_cond2_after'].mean():.3f} / {flags['max_cond2_after'].max():.3f}")
    print(sep)  # noqa: ascii-safe
