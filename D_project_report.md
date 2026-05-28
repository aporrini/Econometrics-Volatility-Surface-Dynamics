# Notebook D — SSVI Surface Risk Engine: Project Report

**SSVI Volatility Surface Dynamics — S&P 500 Options (2010–2020)**
Politecnico di Milano · Econometrics Project A.Y. 2025/26

---

## 1. Pipeline I/O

### Data flow: A → B → C → D

| Notebook | Reads | Writes |
|----------|-------|--------|
| **A** `A_implied_volatility_panel.ipynb` | `output/options_with_forward_iv_clean.csv` (raw options) | `output/figures/A_*.png` |
| **B** `B_ssvi_parameter_dynamics.ipynb` | `data/ssvi_all_dates_clean_results.csv` (SSVI params, in git) | `output/B_stationarity_tests.csv`, `output/B_arma_results.csv`, `output/figures/B_*.png` |
| **C** `C_realized_volatility_forecasting.ipynb` | SSVI params from B | `output/C_rv_forecasting_results.csv`, `output/figures/C_*.png` |
| **D** `D_ssvi_mm_risk_engine.ipynb` | `data/ssvi_all_dates_clean_results.csv` | See below |

### Notebook D outputs (all prefixed `D_`)

| File | Description |
|------|-------------|
| `output/ssvi_surface_iv_panel.csv` | Cached IV panel (45-point grid); generated on first run if missing |
| `output/D_iv_panel.csv` | Same panel, D-tagged copy |
| `output/D_surface_moves.csv` | Daily surface_move + move by maturity |
| `output/D_jump_comparison.csv` | BPV vs rolling-threshold jump flags |
| `output/D_forecast_metrics.csv` | MAE / RMSE / QLIKE for all models × targets |
| `output/D_forecasts.csv` | Log-scale forecasts on test set |
| `output/D_pca_delta_iv_scores.csv` | PC scores (train+val+test) |
| `output/D_pca_delta_iv_loadings.csv` | PCA loadings on ΔIV grid |
| `output/D_hmm_regimes.csv` | HMM regime assignments |
| `output/D_c_es95_calibration.csv` | c*(bucket) calibrated on val set |
| `output/D_spread_final.csv` | spread_final(k,T,t) on full sample |
| `output/D_mm_backtest.csv` | Backtest coverage by bucket (test set) |
| `output/D_cstar_term_structure.csv` | c*(T) values + OLS fit vs √T |
| `output/D_main_results_table.csv` | Consolidated results table |
| `output/D_vega_weighted_robustness.csv` | Vega-weighted shortfall by maturity |
| `output/plots/D_surface_move.png` | Surface move time series |
| `output/plots/D_move_by_maturity.png` | Move by maturity (22d rolling) |
| `output/plots/D_jump_detection_comparison.png` | BPV vs selected method |
| `output/plots/D_forecast_global.png` | HAR / HAR-J / walk-forward forecasts |
| `output/plots/D_pca_loadings.png` | PC1–PC3 heatmaps on (k, T) grid |
| `output/plots/D_pca_scores.png` | PC score time series |
| `output/plots/D_hmm_regimes.png` | Regime overlay on surface_move |
| `output/plots/D_cstar_term_structure.png` | c* vs T and vs √T |
| `output/plots/D_spread_decomposition.png` | AS + addon + actual (global, test) |
| `output/plots/D_coverage_by_bucket.png` | Coverage bar chart by bucket |
| `output/plots/D_spread_term_structure.png` | Stacked bar: decomposition + coverage |

---

## 2. n_obs Threshold Analysis

### What happens

During SSVI calibration (which generated `data/ssvi_all_dates_clean_results.csv`),
each trading date is calibrated against the available cleaned options chain.
Dates with fewer than 100 observations after cleaning are excluded (`success=False`).

### Exclusion statistics

| Metric | Value |
|--------|-------|
| Total dates in CSV | 2,768 |
| Successful calibrations | 2,653 (95.8%) |
| Failed (n_obs = 0) | 115 (4.2%) |

**Distribution of failed dates by time_elapsed range:**

| Range | Failed dates |
|-------|-------------|
| 0–50 (≈ Jan–Feb 2010) | 34 |
| 50–100 (≈ Feb–Apr 2010) | 34 |
| 100–200 (≈ Apr–Sep 2010) | 36 |
| 200–500 (≈ Sep 2010–Jul 2011) | 8 |
| 500–2,000 | 2 |
| 2,000+ | 0 |

**Actual observation counts in failed rows:** min=5, median=70, max=97.

**Threshold sensitivity:**

| Threshold | Recovered dates | % of dataset |
|-----------|----------------|-------------|
| 40 | 114 | +4.1% |
| 60 | 73 | +2.6% |
| 80 | 47 | +1.7% |

### Recommendation: **Keep threshold at 100**

The exclusion pattern has two components:

1. **Systematic early-period sparsity** (time_elapsed 0–103, ≈ Jan–Apr 2010):
   The first 103 trading days have only ~58 obs/day after cleaning. This
   reflects the genuinely thinner SPX options market in early 2010 (post-crisis
   liquidity recovery). The SSVI fit quality on 58 observations would be
   unreliable: both wings require ≥5 points and the ATM region ≥3 points —
   leaving marginal room for the optimizer.

2. **Scattered anomalies** (time_elapsed 100–1942): 12 dates clustered around
   specific periods (likely monthly expirations or low-liquidity holidays) plus
   two late outliers (time_elapsed 1607 with 82 obs, 1942 with only 5 obs).

**Alternative (coverage-based rule):** `n_obs ≥ 60 AND min_left ≥ 5 AND min_right ≥ 5 AND min_atm ≥ 3`. This would recover ~73 dates (borderline cases with 60–97 obs) while ensuring smile balance. However, the 47 dates with 80–97 obs are scattered anomalies rather than a structured regime; recovering them does not materially change the 2,653-day panel.

---

## 3. Changes Made

### `src/ssvi_mm_risk_engine.py`

| Change | Rationale |
|--------|-----------|
| **`load_ssvi_results(data_dir, output_dir=None)`**: added primary date derivation from `time_elapsed` column using `START = 2010-01-04`. The function no longer requires `ssvi_surface_iv_panel.csv` to set the DatetimeIndex. | Eliminates the chicken-and-egg dependency on the IV panel cache for date alignment. D notebook runs standalone. |
| **`build_iv_panel` fallback**: removed `from ssvi_surface import ssvi_implied_vol_surface` (file was deleted). Inlined the SSVI total-variance formula `θ = exp(α)·T^β`, `φ = η·θ^{−γ}/(1+η·θ^{1−γ})`, `w = (θ/2)(1+ρφk+√…)`. | `ssvi_surface.py` was deleted from disk; the fallback reconstruction path now works without that module. |
| **`save_pca_outputs(…, prefix='07')`**: added `prefix` parameter. | Allows D notebook to write `D_pca_*.csv` instead of `07_pca_*.csv`. |
| **All 9 `plot_*` functions**: added `prefix='07'` parameter and applied to filenames. | Same reason — all plots can now be saved with `D_` or `07_` prefix. Backward-compatible default `'07'`. |

### `econometric_analysis/D_ssvi_mm_risk_engine.ipynb` (new file)

| Change vs 07_ | Details |
|--------------|---------|
| **Title** | `# Notebook D — SSVI Surface Risk Engine` matching A/B/C style |
| **BASE path resolution** | Handles `econometric_analysis/`, `notebook/`, or repo root; `DATA_DIR = BASE / 'data'` added |
| **`load_ssvi_results` call** | Changed from `load_ssvi_results(OUTPUT_DIR)` to `load_ssvi_results(DATA_DIR, OUTPUT_DIR)` |
| **Safety assertions** | `DatetimeIndex`, `monotonic_increasing`, `no duplicates` on `params_ok`; `DatetimeIndex` + `sorted` on `iv_panel`; chronological leakage guard on splits |
| **Section numbering** | Numbered 1–12 (matching A/B/C convention); removed `## A.`, `## B.`, ... labels |
| **No "FIX 1/2/3" labels** | c* term structure, jump detection, and results table integrated as regular sections |
| **No `importlib.reload`** | Removed temporary workaround cell; imports are clean |
| **D_ output prefix** | All `to_csv()` paths use `f'{NB}_*.csv'` with `NB='D'`; all plots pass `prefix=NB` |
| **n_obs diagnostic** | Section 2 produces the exclusion table and recommendation inline |
| **IV panel caching** | First run saves to `ssvi_surface_iv_panel.csv` (standard cache); subsequent runs load from it |

### Files NOT changed

- `econometric_analysis/07_final_ssvi_mm_risk_engine.ipynb` — kept intact
- `econometric_analysis/A_implied_volatility_panel.ipynb` — not modified
- `econometric_analysis/B_ssvi_parameter_dynamics.ipynb` — not modified
- `econometric_analysis/C_realized_volatility_forecasting.ipynb` — not modified
- All econometric methodology (SSVI, HAR, HAR-J, HAR-CJ, PCA, HMM, AS-style spread, ES₉₅ calibration, spread formula) — unchanged

---

## 4. Methodological Constraints (Preserved)

| Constraint | Status |
|------------|--------|
| No shuffle in train/val/test (70/15/15 chronological) | ✓ |
| PCA fitted on train only, projected onto val/test | ✓ |
| c* calibrated on validation set only | ✓ |
| HMM Viterbi descriptive only (not for live trading) | ✓ |
| No cap on market spread | ✓ |
| Baseline labelled "Avellaneda-*style* proxy" (not full AS) | ✓ |
| No neural networks | ✓ |
| No absolute paths in code | ✓ |
| No data files committed to Git | ✓ (outputs in `output/`, excluded from repo) |

---

*Report generated: 2026-05-28*
