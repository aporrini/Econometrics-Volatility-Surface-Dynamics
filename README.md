# SSVI Volatility Surface Dynamics — S&P 500 Options (2010–2020)

**Politecnico di Milano — Econometrics Project (A.Y. 2025/26)**  
Authors: Alessio Porrini, Marco Amarilli, Camilla Introzzi, Christian Frigerio

---

## Overview

This repository contains the full analytical pipeline for studying SSVI (Surface Stochastic Volatility Inspired) implied volatility dynamics and realized volatility forecasting using **SPX daily European options from 2010 to 2020** (~2,780 trading days, ~1.8 million Black-76 IV observations).

The dataset spans multiple distinct market regimes:
- Post–financial-crisis recovery (2010–2012)
- Low-volatility bull market (2013–2017)
- **Volmageddon event** — February 5, 2018 (structural break in volatility dynamics)
- Late-cycle / pre-COVID period (2019)
- COVID-19 market crash and recovery (2020)

**Data window**: 2010-01-01 → 2020-12-31.  
**Reference date** for `Time Elapsed`: `2010-01-04` (first trading day of 2010).

---

## Prerequisites

The notebooks in this repository (NB03–NB11) require pre-built input files placed in `output/`:

| File | Description |
|------|-------------|
| `FullOptionDatasetEngineered.csv` | Engineered option dataset (all columns) |
| `options_with_forward_iv_clean.csv` | Forward prices + Black-76 implied volatility |
| `ssvi_all_dates_clean_results.csv` | Daily SSVI calibration results |

**Engineered dataset schema:**

| Column | Description |
|--------|-------------|
| `Time Elapsed` | Calendar days from `2010-01-04` |
| `OPT STRIKE PRICE` | Strike price |
| `OPEN INTEREST` | Open interest |
| `Instrument` | Option ticker (e.g. `SPX06111100C`) |
| `Mid Price` | (Ask + Bid) / 2 |
| `Liquidity Factor` | (Ask − Bid) / Mid |
| `OptionType` | 1 = Call, −1 = Put |
| `rate1month` … `rate3year` | US Treasury zero-coupon rates (%, from FRED) |
| `sp500` | Implied forward extracted via put-call parity |
| `Moneyness` | log(K / SP500) |
| `TTE` | Time to expiry in years |

**Dataset scale:**

| Stage | Count |
|-------|-------|
| Raw rows loaded (2,769 files) | 3,331,302 |
| After quality filters | 2,324,008 |
| Black-76 IV retained | 1,818,064 (78.2%) |
| SSVI calibration success | 2,666 / 2,768 days (96.3%) |

**Quality filters applied to raw data:**

| Filter | Threshold |
|--------|-----------|
| Mid price | > $0.05 |
| Time to expiry | > 7 days |
| Liquidity factor (bid-ask spread / mid) | ≤ 0.30 |
| Log-moneyness log(K/S) | ∈ [−0.40, 0.30] |

---

## Notebook Structure

Run in order: **03 → 04 → 04.5 → 05 → 05.5 → 06 → 07 → 08 → 08.5 → 09 → 10 → 11**

| # | Notebook | Description | Category |
|---|----------|-------------|----------|
| 03 | `03_panel_iv_regression.ipynb` | Panel OLS/FE/RE + full Gauss-Markov diagnostics | **Core Econometrics** |
| 04 | `04_arma_var_cointegration.ipynb` | ARMA/ARMAX + EGARCH + Conformal Prediction + VAR + Cointegration | **Core Econometrics** |
| 04.5 | `04_5_vecm_rolling_cointegration.ipynb` | Rolling Johansen cointegration (500-day window): rank=1 in 32.1% of windows (stress regimes); global rank=2 with full dataset → VECM globally misspecified; regime-conditional EG p≈0.05 | EXTRA |
| 05 | `05_ssvi_parameter_forecasting.ipynb` | Triple Battle: Sticky vs AR-GARCH vs ARMAX at h=1,5,20 | Core + EXTRA |
| 05.5 | `05_5_ssvi_surface_pca_har.ipynb` | PCA on 100-point SSVI surface grid: 2 PCs explain 99% (PC1=Level 96.65%, PC2=Term structure 2.43%). HAR on PC scores: R²_OOS=+0.763/+0.876/+0.805 at h=1/5/20. Nelson-Siegel analogy for vol surfaces. | EXTRA |
| 06 | `06_ssvi_feature_engineering.ipynb` | HAR-RV components, SSVI features, forecasting dataset | EXTRA |
| 07 | `07_rv_forecasting_core.ipynb` | RV20 forecasting: Ridge, Lasso, HAR-RV benchmark | EXTRA |
| 08 | `08_rv_forecasting_extensions.ipynb` | Multi-horizon log-HAR (h=1,5,20), regime-augmented HAR (F3_VIX: R²=0.861 at h=5), VVIX×ρ test, rolling OOS, Generic HAR (ATM_SSVI replaces VIX: R²=0.837, portable to any asset), SSVI-motivated models M1–M4 (**M4 wins: R²=0.868 at h=20, DM p<0.0001**), beta extensions M5/M6 | EXTRA |
| 08.5 | `08_5_rv_features_nb0607.ipynb` | NB06-07 feature additions to M4: log_ATM×max_cond1 (near-arb stress interaction) improves h=5 (DM=−8.75***, R²=0.847); log_rmse_iv hurts (DM=+7.19***) | EXTRA |
| 09 | `09_frontier_analyses.ipynb` | Frontier analyses: Volmageddon/COVID structural breaks, Granger causality, Andres et al. benchmark | EXTRA |
| 10 | `10_novelties.ipynb` | Novel findings summary (6 findings): η negative sign, β_dev OOS gain, SSVI portability, Heston↔SSVI↔HAR mapping, smooth>discrete regime, SSVI Δ-params near-white-noise | EXTRA |
| 11 | `11_quantum_ssvi_parameter_forecasting_qiskit.ipynb` | Can quantum feature maps model nonlinear SSVI dynamics? Naive vs GradientBoosting vs Quantum Kernel SVR (ZZFeatureMap + FidelityQuantumKernel). Result: SSVI Δ-params near-white-noise; only Δγ marginally predictable (MSE_ratio=0.95). Qiskit graceful skip if not installed. | EXTRA |

---

## Key Results Summary (NB03–NB09)

### NB03 — Panel IV Regression
- R² progression: Pooled OLS 0.63 → Day FE 0.73 → Day+Maturity FE **0.77** → Surface Cell FE 0.81
- Day FE implemented via within-transformation to avoid the 37 GB design matrix
- RESET test rejects the linear specification → log(IV) or IV²(moneyness) recommended

### NB04 — ARMA / VAR / Cointegration on SSVI Parameters
**Integration order** (2,666 observations, 10-year panel — high statistical power):
- I(1): **alpha, eta** — share a stochastic trend
- I(0): **beta, rho, gamma** — mean-revert over the full window

**Correct specification**: Δalpha, Δeta (I(1) → first-differenced); beta, rho, gamma (I(0) → levels).

**BIC-optimal ARMA orders:**

| Series | Order | Ljung-Box adequate? |
|--------|-------|---------------------|
| d_alpha | ARMA(1,0) | borderline (p=0.008) |
| rho     | ARMA(1,3) | borderline (p=0.002) |
| gamma   | ARMA(1,4) | ✓ (p=0.651) |

**OOS MSE ratios (model / random-walk baseline):**

| Series | ARMA | ARMAX |
|--------|------|-------|
| d_alpha | 0.986 ✓ | 0.986 ✓ |
| rho     | 1.089 | 1.161 |
| gamma   | 0.886 ✓ | 0.879 ✓ |

**Prediction interval coverage (80% target):**  
GARCH(1,1): d_alpha 82.2%, rho 69.0%, gamma 81.6%  
Conformal PI: rho 81.4% ✓, gamma 87.2% ✓; d_alpha 66.6% (exchangeability violated for time series)

**Cointegration (alpha–eta system)**: Johansen rank=2, Engle-Granger p=0.006.  
VECM estimated on the alpha–eta subsystem; loading matrix shows speed of adjustment to long-run equilibrium.

---

## Output Files

```
output/
  FullOptionDatasetEngineered.csv     # engineered option dataset (all columns)
  _cache_rates.csv                    # Treasury rate cache (FRED)
  options_with_forward_iv_clean.csv   # forward prices + Black-76 IV
  ssvi_all_dates_clean_results.csv    # daily SSVI calibrations
  ssvi_forecasting_dataset.csv        # supervised forecasting dataset (nb06)
  04a_core_model_results.csv          # RV forecasting core results (nb07)
  04b_all_model_results.csv           # extended results (nb08)
  shap_values_test.csv                # SHAP values (nb08)
  violation_rate_summary.csv          # no-arb violation rates (nb05)

output/plots/
  (plots generated by notebooks 03–09)
```

---

## Reproducibility

```
# Required packages
pip install pandas-datareader

# Run in order
jupyter nbconvert --to notebook --execute "Notebooks Analysis/03_panel_iv_regression.ipynb"
jupyter nbconvert --to notebook --execute "Notebooks Analysis/04_arma_var_cointegration.ipynb"
# ... continue through NB11
```

> **Note on runtime**: with ~2,780 trading days the notebooks are substantially more compute-intensive than short-window analyses. Expect 30–90 minutes total for a full pipeline run.

---

## References

- **Gatheral, J. & Jacquier, A. (2014).** Arbitrage-free SVI volatility surfaces. *Quantitative Finance*, 14(1), 59–71.
- **Corsi, F. (2009).** A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
- **Andersen, T., Bollerslev, T., Diebold, F. & Labys, P. (2003).** Modeling and forecasting realized volatility. *Econometrica*, 71(2), 579–625.
- **Bollerslev, T. & Wooldridge, J. (1992).** Quasi-maximum likelihood estimation and inference in dynamic models with time-varying covariances. *Econometric Reviews*, 11(2), 143–172.
- **Johansen, S. (1991).** Estimation and hypothesis testing of cointegration vectors in Gaussian VAR models. *Econometrica*, 59(6), 1551–1580.
- **Engle, R. & Granger, C. (1987).** Co-integration and error correction: representation, estimation, and testing. *Econometrica*, 55(2), 251–276.
- **Andrès, H., Boumezoued, A. & Jourdain, B. (2025).** The implied volatility surface (also) is path-dependent. Working paper (arXiv v3, 2025).
- **Fontana, M., Zeni, G. & Vantini, S. (2023).** Conformal Prediction: a Unified Review of Theory and New Challenges. *Bernoulli*, 29(1), 1–23.
- **Diebold, F. & Mariano, R. (1995).** Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
