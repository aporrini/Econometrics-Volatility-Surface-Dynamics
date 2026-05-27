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

### Notebook C — Self-Contained Forecasting Framework (Primary)

`econometric_analysis/C_realized_volatility_forecasting.ipynb` is the **primary deliverable** for the realized volatility forecasting component. It consolidates the full multi-horizon forecasting pipeline into a single, publication-ready notebook.

| Section | Description |
|---------|-------------|
| 1. Data Loading | SSVI parameters from GitHub; SP500 from Yahoo Finance / FRED fallback; VIX from FRED |
| 2. Realized Volatility | Daily log-RV proxy; multi-horizon targets h ∈ {1, 5, 20} |
| 3. Feature Engineering | HAR components, SSVI levels, ATM IV, centered stress interactions |
| 4. Evaluation Framework | OOS R², DM-HLN test (Newey-West HAC + HLN small-sample correction) |
| 5. Multi-Horizon Evaluation | 8 models, 80/20 single split |
| 6a. Collinearity & Feature Selection | VIF analysis; significance-based pruning (p < 0.10); centered lambda fix |
| 6b. Expanding Window | Fully recursive OOS with min 60% training; 6 models |
| 7. Visualizations | R²_OOS heatmap, forecast vs actual, lambda stress, MSE ratio plots |
| 8. Key Findings | Economic interpretation, regime dynamics |
| 9. Model Formulas | Complete mathematical specification of all models |

**Final model set:**

| Model | Features | Best R²_OOS |
|-------|----------|-------------|
| Naive | RW | 0.000 |
| HAR | 3 (d/w/m RV) | ~0.82 (h=5) |
| HAR+VIX | 4 | ~0.82 (h=5) |
| F3_ATM | 4, portable | ~0.84 (h=5) |
| HAR_rhoJ | 4, jump proxy | ~0.82 (h=5) |
| M1 | 4 (HAR+α) | ~0.83 (h=5) |
| **M4_smooth** | **9** (HAR + {α,ρ,γ,λ̃,α·λ̃,ρ·λ̃}) | **~0.87 (h=20)** |
| **M4+int** | **12** (M4_smooth + {β,η,η·λ̃,ATM·κ}) | **~0.87 (h=20)** |

Key methodological contributions: (1) centered lambda interaction eliminates VIF > 6000; (2) significance-based feature selection at h=5 applied uniformly across horizons; (3) fully recursive expanding window with numpy lstsq for 50× speedup.


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
