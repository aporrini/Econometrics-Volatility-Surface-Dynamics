"""
forward_curve.py
================
Model-free implied forward estimation for SPX options via put-call parity.

Why forward pricing for SPX?
-----------------------------
The S&P 500 pays dividends (~1.5–2 % annually).  Pricing options with the
raw spot S and risk-free rate r alone implicitly sets the dividend yield to
zero.  This distorts:

  1. Moneyness: log(K/S) overestimates (underestimates) moneyness for calls
     (puts) relative to the true log(K/F).  The error grows with TTE.

  2. Right-wing IV: the intrinsic-value lower bound for calls is
     max(S − K·exp(−rT), 0) instead of the correct DF·max(F−K, 0).
     Call prices near intrinsic get assigned NaN instead of a finite IV.

  3. SSVI calibration: mixing log(K/S) across maturities introduces a
     bias proportional to dividend_yield × TTE that artificially tilts the
     fitted skew.

Put-call parity recovers F model-free (no dividend assumption needed):

    C − P = exp(−rT) · (F − K)    →    F = K + exp(rT) · (C − P)

We aggregate over near-ATM pairs per (date, TTE) using an OI-weighted
median to obtain a robust, noise-resistant forward estimate.

Functions
---------
interpolate_rate(row)            → decimal risk-free rate
compute_pair_forward(df)         → one row per call/put pair
estimate_forward_curve(pairs)    → one row per (date, TTE)
attach_forward_to_options(df, fc) → adds forward + forward_moneyness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RATE_MATURITIES = np.array([1 / 12, 3 / 12, 6 / 12, 1.0, 2.0, 3.0])
_RATE_COLS = [
    "rate1month", "rate3month", "rate6month",
    "rate1year",  "rate2year",  "rate3year",
]
_MERGE_COLS = ["Time Elapsed", "TTE", "OPT STRIKE PRICE"]


# ---------------------------------------------------------------------------
# Rate interpolation
# ---------------------------------------------------------------------------

def interpolate_rate(row) -> float:
    """Return the linearly-interpolated risk-free rate for one option row.

    Reads rate1month … rate3year (percentage form: 1.20 = 1.20 %) and
    interpolates to TTE.  Returns the rate as a decimal.
    Returns np.nan if TTE is invalid or any rate is missing.
    """
    T = row["TTE"]
    if not np.isfinite(T) or T <= 0:
        return np.nan
    rates = np.array([row[c] for c in _RATE_COLS], dtype=float)
    if np.any(~np.isfinite(rates)):
        return np.nan
    return float(np.interp(T, _RATE_MATURITIES, rates / 100.0))


# ---------------------------------------------------------------------------
# Step 1: matched call/put pairs
# ---------------------------------------------------------------------------

def compute_pair_forward(
    df: pd.DataFrame,
    fuzzy_tte_tol: float = 0.0,
) -> pd.DataFrame:
    """Compute the implied forward from put-call parity for matched pairs.

    For each (Time Elapsed, TTE, OPT STRIKE PRICE) triple where a call
    and a put both exist:

        F = K + exp(r * T) * (C_mid − P_mid)

    Parameters
    ----------
    df            : options DataFrame; must contain OptionType, Mid Price,
                    and the merge columns (Time Elapsed, TTE, OPT STRIKE PRICE).
    fuzzy_tte_tol : tolerance for fuzzy TTE matching (default 0 = exact match).
                    When > 0, calls and puts whose TTE differs by at most
                    `fuzzy_tte_tol` years are also paired; the representative
                    TTE of each fuzzy pair is set to the average of the two.
                    Useful for sub-day rounding differences (~1e-5 ≈ 5 min).

    Returns
    -------
    DataFrame with one row per matched (call, put) pair:
        Time Elapsed, TTE, OPT STRIKE PRICE,
        forward_pair, r, abs_moneyness_spot, liquidity_pair, oi_pair
    """
    for col in _MERGE_COLS + ["Mid Price", "OptionType"]:
        if col not in df.columns:
            raise KeyError(f"Required column missing: {col!r}")

    calls = df[df["OptionType"] == 1]
    puts  = df[df["OptionType"] == -1]

    # ── Call sub-frame: keep rate cols + ancillary for forward computation ──
    call_want = list(dict.fromkeys(
        _MERGE_COLS + _RATE_COLS
        + ["Mid Price", "Liquidity Factor", "OPEN INTEREST"]
        + [c for c in ("sp500", "Moneyness") if c in df.columns]
    ))
    calls_sub = calls[[c for c in call_want if c in calls.columns]].copy()
    calls_sub = calls_sub.rename(columns={
        "Mid Price":        "mid_call",
        "Liquidity Factor": "lf_call",
        "OPEN INTEREST":    "oi_call",
    })

    # ── Put sub-frame: only price + liquidity + OI ──────────────────────────
    put_want  = _MERGE_COLS + ["Mid Price", "Liquidity Factor", "OPEN INTEREST"]
    puts_sub  = puts[[c for c in put_want if c in puts.columns]].copy()
    puts_sub  = puts_sub.rename(columns={
        "Mid Price":        "mid_put",
        "Liquidity Factor": "lf_put",
        "OPEN INTEREST":    "oi_put",
    })

    # ── Exact merge on (date, TTE, strike) ──────────────────────────────────
    pairs_exact = calls_sub.merge(puts_sub, on=_MERGE_COLS, how="inner")
    n_exact = len(pairs_exact)

    if fuzzy_tte_tol > 0:
        # Fuzzy merge: match on (date, strike); filter by |TTE_call - TTE_put|
        loose_cols = ["Time Elapsed", "OPT STRIKE PRICE"]
        calls_tmp = calls_sub.rename(columns={"TTE": "_TTE_c"})
        puts_tmp  = puts_sub.rename(columns={"TTE": "_TTE_p"})

        candidates = calls_tmp.merge(puts_tmp, on=loose_cols, how="inner")
        candidates["_tte_diff"] = (candidates["_TTE_c"] - candidates["_TTE_p"]).abs()
        candidates = candidates[candidates["_tte_diff"] <= fuzzy_tte_tol].copy()

        # Per (date, TTE_call, strike): keep closest put TTE
        candidates = candidates.sort_values("_tte_diff")
        candidates = candidates.drop_duplicates(
            subset=["Time Elapsed", "_TTE_c", "OPT STRIKE PRICE"], keep="first"
        )
        # Per (date, TTE_put, strike): keep closest call TTE
        candidates = candidates.drop_duplicates(
            subset=["Time Elapsed", "_TTE_p", "OPT STRIKE PRICE"], keep="first"
        )

        # Representative TTE = average of the two (exact pairs get avg of equal vals)
        candidates["TTE"] = 0.5 * (candidates["_TTE_c"] + candidates["_TTE_p"])
        candidates = candidates.drop(columns=["_TTE_c", "_TTE_p", "_tte_diff"])

        pairs = candidates
        n_fuzzy_extra = len(pairs) - n_exact
        if n_fuzzy_extra > 0:
            print(f"Fuzzy TTE matching (tol={fuzzy_tte_tol:.0e}): "
                  f"{n_fuzzy_extra:,} additional pairs recovered "
                  f"({n_exact:,} exact + {n_fuzzy_extra:,} fuzzy = {len(pairs):,} total).")
    else:
        pairs = pairs_exact

    if len(pairs) == 0:
        return pd.DataFrame(columns=_MERGE_COLS + [
            "forward_pair", "r", "abs_moneyness_spot", "liquidity_pair", "oi_pair"
        ])

    # ── Forward via put-call parity ─────────────────────────────────────────
    pairs["r"] = pairs.apply(interpolate_rate, axis=1)

    K  = pairs["OPT STRIKE PRICE"].to_numpy(float)
    T  = pairs["TTE"].to_numpy(float)
    r  = pairs["r"].to_numpy(float)
    C  = pairs["mid_call"].to_numpy(float)
    P  = pairs["mid_put"].to_numpy(float)

    rT = np.where(np.isfinite(r) & np.isfinite(T), r * T, np.nan)
    pairs["forward_pair"] = K + np.exp(rT) * (C - P)

    # ── Quality metrics for ATM selection and weighting ─────────────────────
    if "sp500" in pairs.columns:
        S = pairs["sp500"].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            pairs["abs_moneyness_spot"] = np.abs(
                np.where((S > 0) & (K > 0), np.log(K / S), np.nan)
            )
    elif "Moneyness" in pairs.columns:
        pairs["abs_moneyness_spot"] = np.abs(pairs["Moneyness"])
    else:
        pairs["abs_moneyness_spot"] = np.nan

    lf_c = pairs["lf_call"].fillna(0).to_numpy(float) if "lf_call" in pairs.columns else np.zeros(len(pairs))
    lf_p = pairs["lf_put"].fillna(0).to_numpy(float)  if "lf_put"  in pairs.columns else np.zeros(len(pairs))
    oi_c = pairs["oi_call"].fillna(0).to_numpy(float) if "oi_call" in pairs.columns else np.zeros(len(pairs))
    oi_p = pairs["oi_put"].fillna(0).to_numpy(float)  if "oi_put"  in pairs.columns else np.zeros(len(pairs))

    pairs["liquidity_pair"] = np.maximum(lf_c, lf_p)
    pairs["oi_pair"]        = oi_c + oi_p

    # Remove implausible forwards
    fwd   = pairs["forward_pair"].to_numpy(float)
    pairs = pairs[np.isfinite(fwd) & (fwd > 0)].reset_index(drop=True)

    keep = _MERGE_COLS + [
        "forward_pair", "r", "abs_moneyness_spot", "liquidity_pair", "oi_pair"
    ]
    return pairs[[c for c in keep if c in pairs.columns]]


# ---------------------------------------------------------------------------
# Step 2: robust forward per (date, TTE)
# ---------------------------------------------------------------------------

def estimate_forward_curve(
    df_pairs: pd.DataFrame,
    atm_band: float = 0.05,
) -> pd.DataFrame:
    """OI-weighted median forward per (Time Elapsed, TTE).

    Prefers near-ATM pairs (|log(K/S)| ≤ atm_band) where put-call parity
    is most reliable.  Falls back to all pairs when no ATM pairs exist.

    Parameters
    ----------
    df_pairs : output of compute_pair_forward()
    atm_band : log-moneyness band for ATM selection (default 0.05 ≈ ±5 %)

    Returns
    -------
    DataFrame with columns:
        Time Elapsed, TTE, forward, n_pairs, forward_std
    """
    if len(df_pairs) == 0:
        return pd.DataFrame(
            columns=["Time Elapsed", "TTE", "forward", "n_pairs", "forward_std"]
        )

    def _oi_median(fwds: np.ndarray, oi: np.ndarray) -> float:
        oi  = np.maximum(oi, 1.0)
        oi  = oi / oi.sum()
        idx = np.argsort(fwds)
        pos = int(np.searchsorted(np.cumsum(oi[idx]), 0.5))
        return float(fwds[idx[min(pos, len(fwds) - 1)]])

    has_atm = "abs_moneyness_spot" in df_pairs.columns
    has_oi  = "oi_pair" in df_pairs.columns

    records = []
    for (te, tte), all_grp in df_pairs.groupby(["Time Elapsed", "TTE"]):
        # Prefer ATM; fall back to all pairs if none qualify
        grp = (
            all_grp[all_grp["abs_moneyness_spot"].fillna(np.inf) <= atm_band]
            if has_atm else all_grp
        )
        if len(grp) == 0:
            grp = all_grp

        fwds = grp["forward_pair"].to_numpy(float)
        fwds = fwds[np.isfinite(fwds)]
        if len(fwds) == 0:
            continue

        oi = grp["oi_pair"].fillna(1.0).to_numpy(float) if has_oi else np.ones(len(fwds))

        records.append({
            "Time Elapsed": te,
            "TTE":          tte,
            "forward":      _oi_median(fwds, oi),
            "n_pairs":      len(fwds),
            "forward_std":  float(np.std(fwds)) if len(fwds) > 1 else 0.0,
        })

    return (
        pd.DataFrame(records)
        .sort_values(["Time Elapsed", "TTE"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Step 3: attach to options DataFrame
# ---------------------------------------------------------------------------

def attach_forward_to_options(
    df: pd.DataFrame,
    forward_curve: pd.DataFrame,
) -> pd.DataFrame:
    """Merge the forward curve onto the options DataFrame.

    Adds:
        forward           — implied forward price F(date, TTE)
        forward_moneyness — log(K / F)

    Rows with no matching (date, TTE) forward receive NaN in both columns.
    """
    df = df.copy()
    fc = forward_curve[["Time Elapsed", "TTE", "forward"]].copy()
    df = df.merge(fc, on=["Time Elapsed", "TTE"], how="left")

    K = df["OPT STRIKE PRICE"].to_numpy(float)
    F = df["forward"].to_numpy(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        df["forward_moneyness"] = np.where(
            np.isfinite(F) & (F > 0) & np.isfinite(K) & (K > 0),
            np.log(K / F),
            np.nan,
        )

    cov = df["forward"].notna().mean() * 100
    print(f"Forward attached: {cov:.1f}% of rows have a valid forward.")
    return df


# ---------------------------------------------------------------------------
# Step 4: forward curve quality flags
# ---------------------------------------------------------------------------

def flag_forward_nodes(
    forward_curve: pd.DataFrame,
    df: pd.DataFrame,
    min_pairs: int = 5,
    max_std: float = 5.0,
) -> pd.DataFrame:
    """Annotate forward curve nodes with quality flags and forward/spot ratio.

    Parameters
    ----------
    forward_curve : output of estimate_forward_curve()
    df            : options DataFrame containing 'sp500' (used for spot price)
    min_pairs     : flag nodes with n_pairs < min_pairs  (default 5)
    max_std       : flag nodes with forward_std > max_std (default 5.0)

    Returns
    -------
    Copy of forward_curve with added columns:
        spot        – median sp500 for that (date, TTE)
        ratio       – forward / spot
        n_pairs_ok  – bool: n_pairs >= min_pairs
        std_ok      – bool: forward_std <= max_std
        ok          – bool: both flags pass
    """
    spot = (
        df.groupby(["Time Elapsed", "TTE"])["sp500"]
        .median()
        .reset_index()
        .rename(columns={"sp500": "spot"})
    )
    fc = forward_curve.merge(spot, on=["Time Elapsed", "TTE"], how="left")
    with np.errstate(invalid="ignore", divide="ignore"):
        fc["ratio"] = np.where(
            fc["spot"].notna() & (fc["spot"] > 0),
            fc["forward"] / fc["spot"],
            np.nan,
        )
    fc["n_pairs_ok"] = fc["n_pairs"] >= min_pairs
    fc["std_ok"]     = fc["forward_std"] <= max_std
    fc["ok"]         = fc["n_pairs_ok"] & fc["std_ok"]

    n_bad = (~fc["ok"]).sum()
    if n_bad > 0:
        print(f"Forward curve: {n_bad}/{len(fc)} nodes flagged "
              f"({n_bad/len(fc)*100:.1f}%) — "
              f"{(~fc['n_pairs_ok']).sum()} low-pair, "
              f"{(~fc['std_ok']).sum()} high-std.")
    else:
        print(f"Forward curve: all {len(fc)} nodes pass quality checks.")
    return fc


def plot_forward_diagnostics(
    fc_flagged: pd.DataFrame,
    sample_dates: list | None = None,
    figsize_ratio: tuple = (14, 4),
    figsize_time: tuple = (14, 8),
) -> None:
    """Three diagnostic plots for the forward curve.

    Plot A (2 panels): forward/spot ratio vs TTE for selected dates.
    Plot B (2 panels): forward_std and n_pairs over calendar time.

    Parameters
    ----------
    fc_flagged   : output of flag_forward_nodes()
    sample_dates : list of Time Elapsed values to show in plot A;
                   default = first 6 unique dates
    """
    if sample_dates is None:
        sample_dates = sorted(fc_flagged["Time Elapsed"].unique())[:6]

    # ── Plot A: forward/spot ratio vs TTE ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=figsize_ratio, constrained_layout=True)
    fig.suptitle("Forward / Spot ratio vs TTE (selected dates)", fontsize=12)

    for te in sample_dates:
        sub = fc_flagged[fc_flagged["Time Elapsed"] == te].sort_values("TTE")
        axes[0].plot(sub["TTE"] * 365, sub["ratio"], marker="o", ms=4,
                     lw=1.4, label=f"date {te}")
        axes[1].plot(sub["TTE"] * 365, sub["forward_std"], marker="o", ms=4,
                     lw=1.4, label=f"date {te}")

    axes[0].axhline(1.0, color="black", lw=0.8, ls="--", label="F/S = 1")
    axes[0].set_xlabel("TTE (days)")
    axes[0].set_ylabel("F / S")
    axes[0].set_title("Forward / Spot ratio")
    axes[0].legend(fontsize=8)

    axes[1].axhline(5.0, color="crimson", lw=0.8, ls="--", label="std limit = 5")
    axes[1].set_xlabel("TTE (days)")
    axes[1].set_ylabel("Forward std (price units)")
    axes[1].set_title("Forward dispersion (forward_std)")
    axes[1].legend(fontsize=8)
    plt.show()

    # ── Plot B: n_pairs and forward_std aggregated over time ─────────────────
    agg = (
        fc_flagged.groupby("Time Elapsed")
        .agg(
            median_n_pairs   = ("n_pairs",      "median"),
            median_fwd_std   = ("forward_std",  "median"),
            pct_bad          = ("ok",           lambda x: (~x).mean() * 100),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(2, 1, figsize=figsize_time, sharex=True,
                             constrained_layout=True)
    fig.suptitle("Forward curve quality over time", fontsize=12)

    axes[0].plot(agg["Time Elapsed"], agg["median_n_pairs"], lw=1.4,
                 color="steelblue", label="Median n_pairs per node")
    axes[0].axhline(5.0, color="crimson", lw=0.8, ls="--", label="min_pairs = 5")
    axes[0].set_ylabel("n_pairs")
    axes[0].legend(fontsize=9)

    ax2 = axes[0].twinx()
    ax2.plot(agg["Time Elapsed"], agg["pct_bad"], lw=1.2, color="darkorange",
             ls="--", alpha=0.7, label="% bad nodes")
    ax2.set_ylabel("% flagged nodes", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    axes[1].plot(agg["Time Elapsed"], agg["median_fwd_std"], lw=1.4,
                 color="purple", label="Median forward_std")
    axes[1].axhline(5.0, color="crimson", lw=0.8, ls="--", label="max_std = 5")
    axes[1].set_xlabel("Time Elapsed (day index)")
    axes[1].set_ylabel("forward_std (price units)")
    axes[1].legend(fontsize=9)

    plt.show()
