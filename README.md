<<<<<<< HEAD
# SSVI Volatility Surface Dynamics — Extended Dataset (2010–2020)

**Politecnico di Milano — Econometrics Project (A.Y. 2024/25)**
Authors: Alessio Porrini et al.

---

## Overview

This folder replicates the full analytical pipeline of `merged_notebook_final`
on a substantially larger dataset: **SPX daily European options from 2010 to 2020**
(approximately 2 780 trading days, vs the ~252-day window in the original pipeline).

The wider time horizon covers multiple distinct market regimes:
- Post–financial-crisis recovery (2010–2012)
- Low-volatility bull market (2013–2017)
- **Volmageddon event** — February 5, 2018 (structural break in volatility dynamics)
- Late-cycle / pre-COVID period (2019)
- COVID-19 market crash and recovery (2020)

**Data source**: `FullOptionData/` — 3 322 per-day CSVs, 2008-01-09 → 2021-03-19.
**Window used**: 2010-01-01 → 2020-12-31 (~2 780 files).
**Reference date** for `Time Elapsed`: `2010-01-04` (first trading day of 2010).

The engineered dataset is saved as `output/FullOptionDatasetEngineered.csv`
with the **identical column schema** as the original `Dati Optionsdsfinal.csv`,
so notebooks 01–09 run unchanged on the new data.

---

## Notebook Structure

Run in order: **00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09**

| # | Notebook | Description | Category |
|---|----------|-------------|----------|
| 00 | `00_data_preparation.ipynb` | **NEW** — Load & merge `FullOptionData/` CSVs, extract implied spot from put-call parity (no external download), fetch Treasury rates (FRED), engineer all features, save `FullOptionDatasetEngineered.csv` | Data prep |
| 01 | `01_iv_dataset_construction.ipynb` | Forward curve estimation, Black-76 IV computation, filtering → `options_with_forward_iv_clean.csv` | Prerequisite |
| 02 | `02_ssvi_calibration.ipynb` | Daily SSVI calibration → `ssvi_all_dates_clean_results.csv` | Prerequisite |
| 03 | `03_panel_iv_regression.ipynb` | Panel OLS/FE/RE + full Gauss-Markov diagnostics | **Core Econometrics** |
| 04 | `04_arma_var_cointegration.ipynb` | ARMA/ARMAX + EGARCH + Conformal Prediction + VAR + Cointegration | **Core Econometrics** |
| 04.5 | `04_5_vecm_rolling_cointegration.ipynb` | **NEW** — Rolling Johansen cointegration (500-day window): rank=1 in 32.1% of windows (stress regimes); global rank=2 with full dataset → VECM globally misspecified; regime-conditional EG p≈0.05 | EXTRA |
| 05 | `05_ssvi_parameter_forecasting.ipynb` | Triple Battle: Sticky vs AR-GARCH vs ARMAX at h=1,5,20 | Core + EXTRA |
| 05.5 | `05_5_ssvi_surface_pca_har.ipynb` | **NEW** — PCA on 100-point SSVI surface grid: 2 PCs explain 99% (PC1=Level 96.65%, PC2=Term structure 2.43%). HAR on PC scores: R²_OOS=+0.763/+0.876/+0.805 at h=1/5/20. Nelson-Siegel analogy for vol surfaces. | EXTRA |
| 06 | `06_ssvi_feature_engineering.ipynb` | HAR-RV components, SSVI features, forecasting dataset | EXTRA |
| 07 | `07_rv_forecasting_core.ipynb` | RV20 forecasting: Ridge, Lasso, HAR-RV benchmark | EXTRA |
| 08 | `08_rv_forecasting_extensions.ipynb` | Multi-horizon log-HAR (h=1,5,20), regime-augmented HAR (F3_VIX: R²=0.861 at h=5), VVIX×ρ test, rolling OOS, Generic HAR (ATM_SSVI replaces VIX: R²=0.837, portability to any asset), SSVI-motivated models M1–M4 (skew-regime, term-structure β, smooth leverage η, **M4 wins: R²=0.868 at h=20, DM p<0.0001**), beta extensions M5/M6 | EXTRA |
| 08.5 | `08_5_rv_features_nb0607.ipynb` | **NEW** — NB06-07 feature additions to M4: log_ATM×max_cond1 (near-arb stress interaction) improves h=5 (DM=−8.75***, R²=0.847); log_rmse_iv hurts (DM=+7.19***). | EXTRA |
| 09 | `09_frontier_analyses.ipynb` | Frontier analyses: regime study, VECM, conformal PI, multi-horizon | EXTRA |
| 10 | `10_novelties.ipynb` | Novel findings summary (6 findings): η negative sign, β_dev OOS gain, SSVI portability, Heston↔SSVI↔HAR mapping, smooth>discrete regime, SSVI Δ-params near-white-noise | EXTRA |
| 11 | `11_quantum_ssvi_parameter_forecasting_qiskit.ipynb` | **Can quantum feature maps model nonlinear dynamics in SSVI surfaces?** Naive vs GradientBoosting vs Quantum Kernel SVR (ZZFeatureMap + FidelityQuantumKernel). Result: SSVI Δ-params near-white-noise; only Δγ marginally predictable (MSE_ratio=0.95). Qiskit graceful skip if not installed. | EXTRA |

---

## Step 0 — Data Preparation (`00_data_preparation.ipynb`)

This notebook is the new entry point, absent in `merged_notebook_final`.

**Inputs**:
- `FullOptionData/*.csv` — 3 322 per-day CSVs with columns:
  `DATE, DATE.1, ASK PRICE, BID PRICE, OPT STRIKE PRICE, OPT EXERCISE PRICE,
  OPEN INTEREST, PRICE HIGH, PRICE LOW, OPENING PRICE, Instrument, exp_date, type`
- FRED (pandas_datareader) — `DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS3`
  (needed by NB01 for Black-76 IV computation)

> **No SP500 download required.** The `sp500` column is filled with the
> options-implied forward extracted from matched call/put pairs via
> put-call parity: $\hat{F} = K + C_{mid} - P_{mid}$ (median per date).
> Notebook 01 then re-derives the rigorous forward with discounting.

**Key steps**:
1. Discover all CSV files in the 2010–2020 window (~2 780 files)
2. Concatenate and clean (handle NaN `OPT STRIKE PRICE` → use `OPT EXERCISE PRICE`)
3. Compute `Mid Price`, `Liquidity Factor`, `OptionType`, `TTE`
4. Extract implied forward from matched call/put pairs (put-call parity, no external data)
5. Fetch Treasury rates (FRED, local cache in `output/_cache_rates.csv`)
6. Apply quality filters (Mid > 0.05, TTE > 7d, LF ≤ 0.30, log(K/S) ∈ [−0.40, 0.30])
7. Assemble final schema and save `output/FullOptionDatasetEngineered.csv`

**Output**: `output/FullOptionDatasetEngineered.csv`

| Column | Description |
|--------|-------------|
| `Time Elapsed` | Calendar days from `2010-01-04` |
| `OPT STRIKE PRICE` | Strike price (coalesced from `OPT EXERCISE PRICE` for early years) |
| `OPEN INTEREST` | Open interest |
| `Instrument` | Option ticker (e.g. `SPX06111100C`) |
| `Mid Price` | (Ask + Bid) / 2 |
| `Liquidity Factor` | (Ask − Bid) / Mid |
| `OptionType` | 1 = Call, −1 = Put |
| `rate1month` … `rate3year` | US Treasury zero-coupon rates (%, from FRED) |
| `sp500` | Implied forward (put-call parity median per date) — no external download |
| `Moneyness` | log(K / SP500) |
| `TTE` | Time to expiry in years |

**Required packages** (not in base conda):
```
pip install pandas-datareader
```

---

## Key Differences from `merged_notebook_final`

| Feature | `merged_notebook_final` | `Notebook_newdata` |
|---------|--------------------------|---------------------|
| Data window | ~252 trading days (mid-2017 → mid-2018) | 2 769 trading days (2010–2020) |
| Reference date | `2017-06-08` | `2010-01-04` |
| Raw data source | `Dati Optionsdsfinal.csv` (pre-built) | `FullOptionData/*.csv` → built by NB00 |
| Engineered file | `Dati Optionsdsfinal.csv` | `FullOptionDatasetEngineered.csv` |
| Raw options | ~203 K | 3 331 302 → 2 324 008 after filtering |
| Black-76 IV rows | ~175 K | 1 818 064 (18 s with vectorised NR) |
| SSVI calibrations | 252 | 2 666 / 2 768 (96.3%) |
| ARMA dataset | ~250 SSVI param observations | 2 666 SSVI param observations |
| Statistical power | Low (short series) | High — 10-year panel |
| Regime diversity | One regime (Volmageddon only) | Post-crisis, bull market, crash, recovery |
| Volmageddon | Centre of dataset | Confirmed at time_elapsed 2954–2977 (near-butterfly-arb cluster in NB02) |

---

## Actual Results Summary (NB00–NB04)

### NB00 — Data preparation
- 3 331 302 raw rows loaded from 2 769 files
- After filters: **2 324 008 rows**, 48 % calls / 52 % puts
- Implied spot (put-call parity): 1 004 – 3 725 (consistent with SP500 2010–2020)
- Treasury rates (FRED): zero missing values after merge

### NB01 — IV construction (vectorised Newton-Raphson, 18 s for 1.9 M rows)
- IV retained: **1 818 064 rows** (78.2 % of raw; 94.0 % of domain-filtered)
- Mean IV = 19.5 %, ATM median IV = 16.2 % (consistent with VIX history)
- ATM call/put IV gap = 0.01 % → put-call parity holds tightly
- Forward / spot ratio: mean = 1.0003 (near-zero carry, post-QE environment)

### NB02 — SSVI calibration
- Success: **2 666 / 2 768 dates** (96.3 %), median RMSE_iv = 0.0081 (< 1 %)
- Parameter medians: α = −3.55, β = 1.22, ρ = −0.75, η = 0.73, γ = 0.54
- ρ ≈ −0.75 confirms strong negative skew (leverage effect); β > 1 → super-linear term structure
- Butterfly-arbitrage-free: 98.1 % of dates
- Volmageddon stress confirmed: near-arbitrage cluster at time_elapsed 2 954–2 977 (Feb 2018)

### NB03 — Panel IV regression
- Day FE and Maturity FE fit via **within-transformation** (avoids 2 774 × 1.8 M = 37 GB matrix)
- R² progression: Pooled OLS 0.63 → Day FE 0.73 → Day+Maturity FE 0.77 → Surface Cell FE 0.81
- `OptionType` absorbed by entity FE (each contract is always a call or put — zero within variation)
- Functional form: RESET test rejects → log(IV) or IV² of moneyness recommended

### NB04 — ARMA / VAR / Cointegration on SSVI parameters
**Stationarity** (2 666 obs, much higher power than original dataset):
- I(1): **alpha, eta** — share a stochastic trend
- I(0): **beta, rho, gamma** — mean-revert over the 10-year window

**Stationarity-correct specification**: d_alpha, d_eta (I(1) → differenced); beta, rho, gamma (I(0) → levels).

**BIC-optimal ARMA orders** (in-sample, full dataset):

| Series | Order | Ljung-Box adequate? |
|--------|-------|---------------------|
| d_alpha | ARMA(1,0) | borderline (p=0.008) |
| d_eta   | *(pending rerun)* | — |
| beta    | *(pending rerun)* | — |
| rho     | ARMA(1,3) | borderline (p=0.002) |
| gamma   | ARMA(1,4) | ✓ (p=0.651) |

**MLE validity**: all series reject Gaussian normality (Shapiro-Wilk) and show ARCH effects
(Ljung-Box on z²) → QMLE applies; GARCH/EGARCH modelling of residuals is warranted.

**OOS forecast (MSE ratio = model / baseline)**:

| Series | ARMA | ARMAX | VAR(2) |
|--------|------|-------|--------|
| d_alpha | 0.986 ✓ | 0.986 ✓ | ~1.007 |
| d_eta   | *(pending)* | *(pending)* | *(pending)* |
| beta    | *(pending)* | *(pending)* | *(pending)* |
| rho     | 1.089 | 1.161 | *(pending)* |
| gamma   | 0.886 ✓ | 0.879 ✓ | *(pending)* |

> **VAR baseline bug corrected**: prior values of 0.001–0.003 for rho/eta/gamma were from a faulty zero baseline (now fixed to random-walk). Realistic MSE ratios pending rerun.

**GARCH(1,1) prediction intervals (80% target)**:
d_alpha 82.2%, d_eta *(pending)*, beta *(pending)*, rho 69.0%, gamma 81.6%

**Conformal PI (split, 80% target)**:
rho 81.4% ✓, gamma 87.2% ✓; d_alpha 66.6% — undershoot (exchangeability violated; coverage approximate). d_eta, beta: pending rerun.

**Cointegration (alpha–eta system)**: Johansen rank=2 (full rank for 2 I(1) variables → strong mean-reversion), Engle-Granger p=0.006. A **VECM** is estimated on the alpha–eta subsystem showing the loading matrix Alpha (speed of adjustment to the long-run equilibrium) alongside a rolling 1-step-ahead OOS comparison vs the random-walk baseline.

---

## Output Files

```
output/
  FullOptionDatasetEngineered.csv     # engineered option dataset (all columns)
  _cache_rates.csv                    # Treasury rate cache (FRED)
  options_with_forward_iv_clean.csv   # forward prices + Black-76 IV (nb01)
  ssvi_all_dates_clean_results.csv    # daily SSVI calibrations (nb02)
  ssvi_forecasting_dataset.csv        # supervised forecasting dataset (nb06)
  04a_core_model_results.csv          # RV forecasting core results (nb07)
  04b_all_model_results.csv           # extended results (nb08)
  shap_values_test.csv                # SHAP values (nb08)
  violation_rate_summary.csv          # no-arb violation rates (nb05)

output/plots/
  00_dataset_overview.png             # options-per-day + SP500 + distributions
  (other plots from notebooks 01–09)
```

---

## Reproducibility

```
# Required packages
pip install pandas-datareader  # no yfinance needed

# Run in order
jupyter nbconvert --to notebook --execute 00_data_preparation.ipynb
jupyter nbconvert --to notebook --execute 01_iv_dataset_construction.ipynb
jupyter nbconvert --to notebook --execute 02_ssvi_calibration.ipynb
# ... continue 03 → 09
```

> **Note on runtime**: with ~2 780 trading days, notebook 02 (SSVI calibration)
> and notebook 06 (feature engineering) will take substantially longer than in
> `merged_notebook_final`. Expect 30–90 minutes total for a full pipeline run.

---

## References

Same as `merged_notebook_final/README.md`, with the addition of:
- **Andrès, H., Boumezoued, A. & Jourdain, B. (2025)**. The implied volatility surface
  (also) is path-dependent. Working paper (arXiv v3, 2025).
- **Fontana, M., Zeni, G. & Vantini, S. (2023)**. Conformal Prediction: a Unified
  Review of Theory and New Challenges. *Bernoulli*, 29(1), 1–23. (Polimi MOX)
=======
# Econometrics-Volatility-Surface-Dynamics
>>>>>>> 28067b75a21d3f89ec68f4d5c68a8c65c17052de
