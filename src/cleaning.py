"""
cleaning.py
===========
Pre-forward cleaning pipeline for S&P 500 options.

Filters (in order):
  1. Strip column whitespace
  2. Remove NaN / Inf in essential columns
  3. Mid Price > min_mid_price      (default 0.05)
  4. TTE > min_ttm                  (default 7/365)
  5. Liquidity Factor <= max_liquidity  (default 0.30)
  6. Spot moneyness in [min_moneyness, max_moneyness]  (default [-0.40, 0.30])

Open interest is NOT filtered — it stays in the dataset as a weighting variable.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _record_step(report: list, step: str, df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
    dates_before = df_before["Time Elapsed"].unique() if "Time Elapsed" in df_before.columns else []
    dates_after  = df_after["Time Elapsed"].unique()  if "Time Elapsed" in df_after.columns  else []
    report.append({
        "step":          step,
        "n_before":      len(df_before),
        "n_after":       len(df_after),
        "dates_before":  len(set(dates_before)),
        "dates_after":   len(set(dates_after)),
        "dates_removed": sorted(set(dates_before) - set(dates_after)),
    })


# ---------------------------------------------------------------------------
# Cleaning report
# ---------------------------------------------------------------------------

def print_cleaning_report(report: list, show_removed_dates: bool = True) -> None:
    """Print a tabular cleaning report."""
    col_w = 52
    header = (
        f"{'Step':<{col_w}} {'Before':>8} {'After':>8} "
        f"{'Removed':>9} {'Removed%':>9} "
        f"{'Dates B':>8} {'Dates A':>8}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for entry in report:
        n_b  = entry["n_before"]
        n_a  = entry["n_after"]
        rem  = n_b - n_a
        pct  = 100.0 * rem / n_b if n_b > 0 else 0.0
        d_b  = entry.get("dates_before", "")
        d_a  = entry.get("dates_after",  "")
        print(
            f"{entry['step']:<{col_w}} {n_b:>8,} {n_a:>8,} "
            f"{rem:>9,} {pct:>8.1f}% "
            f"{d_b:>8} {d_a:>8}"
        )

    print(sep)
    if report:
        total_rem = report[0]["n_before"] - report[-1]["n_after"]
        total_pct = 100.0 * total_rem / report[0]["n_before"] if report[0]["n_before"] > 0 else 0.0
        d0 = report[0].get("dates_before", "")
        dn = report[-1].get("dates_after", "")
        print(
            f"{'TOTAL':<{col_w}} {report[0]['n_before']:>8,} {report[-1]['n_after']:>8,} "
            f"{total_rem:>9,} {total_pct:>8.1f}% "
            f"{d0:>8} {dn:>8}"
        )

    if show_removed_dates:
        removed = {e["step"]: e.get("dates_removed", []) for e in report if e.get("dates_removed")}
        if removed:
            print("\nRemoved dates by step:")
            for step, dates in removed.items():
                print(f"  {step}: {len(dates)} dates")


def print_cleaning_summary(df: pd.DataFrame) -> None:
    """Print call/put/moneyness breakdown of the cleaned dataset."""
    n = len(df)
    calls = (df["OptionType"] == 1).sum()
    puts  = (df["OptionType"] == -1).sum()

    # Use forward_moneyness when available, fall back to spot Moneyness
    k_col = "forward_moneyness" if "forward_moneyness" in df.columns else "Moneyness"
    k     = df[k_col]

    otm_calls   = ((df["OptionType"] ==  1) & (k <  0)).sum()
    right_wing  = ((df["OptionType"] ==  1) & (k >= 0)).sum()   # OTM puts / right wing calls

    print(f"\nCleaning summary  ({k_col} used for moneyness)")
    print(f"  Total retained   : {n:>8,}")
    print(f"  Calls            : {calls:>8,}  ({100*calls/n:.1f}%)")
    print(f"  Puts             : {puts:>8,}  ({100*puts/n:.1f}%)")
    print(f"  OTM calls (k<0)  : {otm_calls:>8,}  ({100*otm_calls/max(calls,1):.1f}% of calls)")
    print(f"  Right-wing (k>=0) : {right_wing:>8,}  ({100*right_wing/max(calls,1):.1f}% of calls)")

    if "Liquidity Factor" in df.columns:
        lf = df["Liquidity Factor"]
        print(f"\n  Liquidity Factor  p25={lf.quantile(0.25):.3f}  "
              f"median={lf.median():.3f}  p75={lf.quantile(0.75):.3f}  "
              f"max={lf.max():.3f}")


# ---------------------------------------------------------------------------
# Individual filter steps
# ---------------------------------------------------------------------------

_ESSENTIAL_COLS = [
    "OPT STRIKE PRICE", "Mid Price", "Liquidity Factor", "OptionType",
    "sp500", "Moneyness", "TTE",
    "rate1month", "rate3month", "rate6month", "rate1year", "rate2year", "rate3year",
]


def remove_missing_and_infinite(
    df: pd.DataFrame,
    report: list,
    essential_cols: list[str] | None = None,
) -> pd.DataFrame:
    cols    = essential_cols if essential_cols is not None else _ESSENTIAL_COLS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    df_out = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols).copy()
    _record_step(report, "NaN / Inf in essential columns", df, df_out)
    return df_out


def remove_non_positive_prices(df: pd.DataFrame, report: list, min_mid_price: float = 0.05) -> pd.DataFrame:
    df_out = df[df["Mid Price"] > min_mid_price].copy()
    _record_step(report, f"Mid Price <= {min_mid_price}", df, df_out)
    return df_out


def remove_short_maturities(df: pd.DataFrame, report: list, min_ttm: float = 7 / 365) -> pd.DataFrame:
    df_out = df[df["TTE"] > min_ttm].copy()
    _record_step(report, f"TTE <= {min_ttm:.4f} ({min_ttm*365:.0f} days)", df, df_out)
    return df_out


def remove_illiquid(df: pd.DataFrame, report: list, max_liquidity: float = 0.30) -> pd.DataFrame:
    df_out = df[df["Liquidity Factor"] <= max_liquidity].copy()
    _record_step(report, f"Liquidity Factor > {max_liquidity}", df, df_out)
    return df_out


def filter_moneyness(
    df: pd.DataFrame,
    report: list,
    min_moneyness: float = -0.40,
    max_moneyness: float =  0.30,
) -> pd.DataFrame:
    df_out = df[
        (df["Moneyness"] >= min_moneyness) & (df["Moneyness"] <= max_moneyness)
    ].copy()
    _record_step(report, f"Moneyness outside [{min_moneyness}, {max_moneyness}]", df, df_out)
    return df_out


def filter_forward_moneyness(
    df: pd.DataFrame,
    report: list,
    min_forward_moneyness: float = -0.30,
    max_forward_moneyness: float =  0.20,
) -> pd.DataFrame:
    """Filter on log(K/F) after the forward curve has been attached.

    Rows where forward_moneyness is NaN (no matching forward node) are also removed.
    Default domain [-0.30, 0.20]: the right wing is truncated at +0.20 because
    deep OTM calls are sparse and noisy, which degrades SSVI calibration.
    """
    if "forward_moneyness" not in df.columns:
        raise KeyError("Column 'forward_moneyness' not found — run attach_forward_to_options() first.")
    df_out = df[
        df["forward_moneyness"].notna()
        & (df["forward_moneyness"] >= min_forward_moneyness)
        & (df["forward_moneyness"] <= max_forward_moneyness)
    ].copy()
    _record_step(
        report,
        f"forward_moneyness outside [{min_forward_moneyness}, {max_forward_moneyness}]",
        df, df_out,
    )
    return df_out


# ---------------------------------------------------------------------------
# Full pre-forward cleaning pipeline
# ---------------------------------------------------------------------------

def clean_options(
    df: pd.DataFrame,
    min_mid_price: float = 0.05,
    min_ttm: float       = 7 / 365,
    max_liquidity: float = 0.30,
    min_moneyness: float = -0.40,
    max_moneyness: float =  0.30,
) -> tuple[pd.DataFrame, list]:
    """Pre-forward cleaning pipeline.

    Steps:
      1. Strip column whitespace
      2. Remove NaN / Inf in essential columns
      3. Mid Price > min_mid_price
      4. TTE > min_ttm
      5. Liquidity Factor <= max_liquidity
      6. Spot moneyness in [min_moneyness, max_moneyness]

    Open interest is kept in the dataset (used as weight, not as filter).

    Returns
    -------
    (cleaned_df, report)
    """
    report: list = []

    df = df.copy()
    df.columns = df.columns.str.strip()

    _record_step(report, "Raw data (before filters)", df, df)

    df = remove_missing_and_infinite(df, report)
    df = remove_non_positive_prices(df, report, min_mid_price=min_mid_price)
    df = remove_short_maturities(df, report, min_ttm=min_ttm)
    df = remove_illiquid(df, report, max_liquidity=max_liquidity)
    df = filter_moneyness(df, report, min_moneyness=min_moneyness, max_moneyness=max_moneyness)

    return df.reset_index(drop=True), report
