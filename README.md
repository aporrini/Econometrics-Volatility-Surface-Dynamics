# SSVI Volatility Surface Dynamics — S&P 500 Options (2010–2020)

**Politecnico di Milano — Econometrics Project (A.Y. 2025/26)**  
Authors: Alessio Porrini, Marco Amarilli, Camilla Introzzi, Christian Frigerio

---

## Overview

Full analytical pipeline for SSVI (Surface Stochastic Volatility Inspired) implied volatility dynamics and realized volatility forecasting using **SPX daily European options**, with calibrated parameters spanning 2010–2025 (~2,652 trading days).

---

## Repository Structure

```
PROGETTO_FINAL/
├── Data/
│   ├── ssvi_all_dates_clean_results.csv   # Daily SSVI calibration (committed)
│   └── no_arbitrage_clean_results.csv
├── Notebooks Data Processing/
│   ├── 00_data_preparation.ipynb          # Raw data loading & quality filters
│   ├── 01_iv_dataset_construction.ipynb   # Black-76 IV via vectorized Newton-Raphson
│   └── 02_ssvi_calibration.ipynb          # Per-day SSVI calibration
├── supplementary/
│   ├── S1_feature_engineering_pipeline.ipynb
│   └── S2_frontier_analyses.ipynb
├── econometric_analysis/
│   ├── A_implied_volatility_panel.ipynb   # Panel IV regression (1.8M obs)
│   ├── B_ssvi_parameter_dynamics.ipynb    # SSVI parameter time-series
│   ├── C_realized_volatility_forecasting.ipynb  # RV forecasting (primary)
│   └── D_ssvi_surface_risk_addon.ipynb    # Surface risk engine + MM spread
└── src/
    ├── ssvi_helpers.py / ssvi_calibration.py / ssvi_mm_risk_engine.py
    ├── black_scholes.py / forward_curve.py / cleaning.py
    └── stats_helpers.py / forecasting_helpers.py / scoring.py
```

---

## Data Processing Notebooks

### `00_data_preparation.ipynb`
Loads ~3.3M raw option observations. Applies quality filters (mid > $0.05, TTE > 7d, bid-ask/mid ≤ 30%, log-moneyness ∈ [−0.40, 0.30]). Output: ~2.3M clean observations.

### `01_iv_dataset_construction.ipynb`
Computes Black-76 IV via vectorized Newton-Raphson. Extracts implied forward prices via put-call parity. Output: ~1.8M IV observations.

### `02_ssvi_calibration.ipynb`
Calibrates SSVI surface per trading day subject to butterfly and calendar no-arbitrage constraints. Writes `Data/ssvi_all_dates_clean_results.csv`. Success rate: **96.3%** (2,653 / 2,768 days), median RMSE_iv < 0.01, zero calendar arbitrage violations.

---

## Primary Analysis Notebooks

### Notebook A — Panel Implied Volatility Regression
`econometric_analysis/A_implied_volatility_panel.ipynb`

Regresses Black-76 IV on option structural features (forward moneyness, TTE, liquidity) across 1.81M observations. Uses within-transformation (time-demeaning) to avoid an explicit 37.5 GB dummy matrix.

Sections: Data loading → EDA → Pooled OLS → Day FE → Day + Maturity FE → Hausman test (FE vs RE) → RESET test → log(IV) robustness → Gauss-Markov diagnostics → Key findings.

---

### Notebook B — SSVI Parameter Dynamics
`econometric_analysis/B_ssvi_parameter_dynamics.ipynb`

**Data:** 2,652 obs (2010–2025). Train: 2,121 obs (through 2022-06-08). Test: 531 obs (2022–2025).

**Sections:**

| Section | Content |
|---------|---------|
| 1. Data Loading | SSVI params (GitHub), VIX/Treasury rates (FRED) |
| 2. ARMA Modelling | BIC-optimal ARMA for 10 series (5 levels + 5 first differences) |
| 3. ARMAX Modelling | Exogenous: Δlog(VIX), VIX z-score |
| 4. VAR Analysis | VAR(2, BIC-optimal) on 5-parameter system |
| 5. Cointegration | Johansen + Engle-Granger for α–η |
| 6. PCA on SSVI Surface | HAR on PC scores (level forecasting) |
| 7. Comprehensive Results | Full horse-race commentary |
| 8. Vol Surface Prediction | Sticky vs ARMA vs VAR surface RMSE |
| Appendix A | Stationarity (ADF + KPSS, 10 series) |
| Appendix B | AR-GARCH prediction intervals |
| Appendix C | Rolling Johansen — regime-conditional cointegration |

**Key actual results:**

- **Stationarity:** All 5 level parameters show conflicting ADF (I(0)) vs KPSS (non-stationary) → classified "Uncertain". First differences are clean I(0). Full parameter descriptives: α mean=−3.45, β mean=1.19, ρ mean=−0.77, η mean=0.75, γ mean=0.54.

- **ARMA OOS (all series):** MSE-ratio = **1.0000**, R²_OOS = **0.0000** across all 10 series. No ARMA specification beats the random walk. BIC-selected orders: α→ARMA(1,0), β/ρ/η/γ→ARMA(1,1).

- **ARMAX OOS:** Same result — MSE-ratio = 1.0000 for all. Despite Granger significance of Δlog(VIX) for α and β (p≈0.017–0.028), the exogenous signal does not translate to OOS improvement.

- **VAR(2) OOS:** Mostly MSE > 1. Only γ achieves MSE-ratio = **0.938** (beats RW slightly).

- **Cointegration (α, η):** Johansen rank = 2; Engle-Granger p = **0.0002*** — cointegrated. VECM OOS: α R²=+0.004 (barely positive), η R²=−0.018 (negative → VECM misspecified for η).

- **Rolling Johansen (500-day windows):** Rank=1 (genuine cointegration) in **29.7%** of windows, rank=2 in 63.6%. Regime-conditional cointegration: EG calm p=0.0004, EG stress p=0.048.

- **PCA on SSVI surface:** 2 PCs explain **99.1%** of variance (PC1=96.34% Level, PC2=2.76% Slope). HAR on PC scores: R²_OOS = **0.926** at h=1, **0.741** at h=5, **0.493** at h=20 (vs naive persistence in surface space).

- **Caching:** Expensive OOS loops cached in `output/cache/`. Set `FORCE_RECOMPUTE = True` to regenerate.

---

### Notebook C — Realized Volatility Forecasting (Primary Deliverable)
`econometric_analysis/C_realized_volatility_forecasting.ipynb`

**Data:** 2,671 obs (2010-05-25 to 2020-12-31). Train: ~2,136 obs (through 2018-11-14). Test: 534 obs (2018-11-15 to 2020-12-30).

Targets $y_t^{(h)} = \frac{1}{h}\sum_{i=1}^h \log\text{RV}_{t+i}$ for $h \in \{1, 5, 20\}$. Evaluation: R²_OOS vs Naive; DM-HLN test (HAC Newey-West + HLN small-sample correction).

**Sections:**

| Section | Content |
|---------|---------|
| 1. Data Loading | SSVI (GitHub), SP500 (Yahoo Finance), VIX (FRED) |
| 2. Realized Volatility | Daily log-RV proxy; multi-horizon targets |
| 3. Feature Engineering | HAR components, SSVI levels, ATM IV, λ-stress interactions |
| 4. Evaluation Framework | R²_OOS definition; DM-HLN test |
| 5. Multi-Horizon Evaluation | 8 models, single 80/20 split |
| 6a. VIF / Feature Selection | VIF analysis; significance-based pruning |
| 6b. Expanding Window | Recursive OOS (numpy lstsq, ~50× speedup) |
| 7. Visualizations | R²_OOS heatmap, forecast comparison, λ_stress, MSE ratio |
| 8. Key Findings | Interpretation of all results |
| Appendix A | Path-dependence test: lagged levels of ρ, γ (Andres et al. 2025) |

**VIF issue (Section 6a):** Uncentered lambda interactions reach VIF = **3,738** (lambda_stress). Centering $\tilde\lambda_t = \lambda_t - \bar\lambda_\text{train}$ resolves the collinearity. Feature selection by OLS significance at h=5 (training fold only) retains {α, ρ, γ} as SSVI levels and {λ̃, α·λ̃, ρ·λ̃} as interactions → M4_smooth (9 features).

**Single-split OOS results (test: 2018–2020):**

| Model | Features | h=1 R²_OOS | h=5 R²_OOS | DM(h=5) | h=20 R²_OOS | DM(h=20) |
|-------|----------|------------|------------|---------|-------------|---------|
| Naive | RW | 0.000 | 0.000 | — | 0.000 | — |
| **HAR** | 3 | **0.465** | **0.846** | — | **0.857** | — |
| HAR+VIX | 4 | 0.489 | 0.832 | −0.59 | 0.864 | +0.53 |
| F3_ATM | 4 | 0.465 | 0.840 | −2.48** | 0.847 | −1.87* |
| HAR_rhoJ | 4 | 0.465 | 0.846 | +0.03 | 0.857 | −0.82 |
| M1 | 4 | 0.466 | 0.842 | −2.51** | 0.849 | −1.92* |
| M4_smooth | 9 | 0.449 | 0.823 | −3.47*** | 0.811 | −2.72*** |
| M4+int | 12 | 0.453 | 0.833 | −2.29** | 0.829 | −1.78* |

DM convention: positive = model beats HAR, negative = model is worse than HAR.

**HAR is the best model.** SSVI-augmented specifications (M4_smooth, M4+int, F3_ATM, M1) are all significantly *worse* than HAR in the single-split evaluation. HAR+VIX is nominally better at h=20 but DM is not significant.

**Expanding window (2016–2020 OOS):**

| Model | h=5 R²_OOS | DM(h=5) | h=20 R²_OOS | DM(h=20) |
|-------|-----------|---------|-------------|---------|
| HAR | 0.853 | — | 0.894 | — |
| **HAR+VIX** | **0.865** | +1.13 | **0.901** | +1.06 |
| F3_ATM | 0.853 | −0.35 | 0.894 | −0.38 |
| M4_smooth | 0.850 | −0.89 | 0.889 | −0.93 |
| M4+int | 0.853 | +0.06 | 0.897 | +0.76 |

In the expanding window, HAR+VIX is the marginal winner but DM is not significant (p≈0.26). All SSVI models converge to HAR performance. HAR remains the correct benchmark.

**The bottom line (Section 8):** The realized vol forecasting task at the test period (2018–2020, including COVID-19 with VIX reaching ~83) is extremely difficult due to distributional shift. HAR's parsimony and direct backward-looking structure outperforms richer SSVI-augmented models in OOS. The structural collinearity of SSVI features (VIF analysis) and distributional shift of regime indicators are the two primary obstacles. F3_ATM (portable, VIX-free) comes closest at h=5 with a small DM advantage over HAR.

---

### Notebook D — SSVI Surface Risk Add-on Engine
`econometric_analysis/D_ssvi_surface_risk_addon.ipynb`

**Data:** 2,652 trading days, 45-point IV grid (9 strikes × 5 maturities). Split: 70/15/15 chronological (no shuffle). surface_move: mean=0.00536 vol-units, std=0.00789.

**Sections:**

| Section | Content |
|---------|---------|
| 1–2 | IV surface reconstruction; calibration exclusion diagnostic |
| 3 | Surface RV: `surface_move(t)` = equal-weighted daily IV grid shift |
| 4 | HAR-J forecasting (HAR + jump indicator, selected from BPV/Method A/B) |
| 5 | Residual-risk add-on: ES₉₅ calibration by (maturity) bucket |
| 5.3 | c*(T) term structure: OLS on √T |
| 6 | Coverage evaluation by bucket |
| 6b | Pointwise surface containment test |
| 7 | Baseline sensitivity (γ parameter) |
| 8 | Summary |
| Appendix A | PCA on ΔIV; HMM regime detection; vega-weighted robustness |

**Key results:**

- **Jump detection:** BPV rate = 31.3% (too loose); selected Method A (q95+2σ) → 1.4% jump rate. HAR-J vs HAR DM is not significant (p=0.959) — jumps add no forecasting improvement.

- **c*(T) term structure:** c*(T) = 5.413 − 3.488√T, R² = **0.941**. c* strictly decreasing: short-dated RV forecasts understate tail risk more severely than long-dated. The √T scaling is consistent with Brownian surface dynamics.

- **Coverage:**

| Bucket | Coverage (spread_AS only) | Coverage (spread_final) |
|--------|--------------------------|------------------------|
| Global | 61.2% | **96.2%** |
| 1M | 28.4% | 97.7% |
| 3M | 54.8% | 97.5% |

The AS-style baseline alone dramatically under-covers (28–61%). The residual-risk add-on (74% of total spread globally) is necessary to reach target coverage.

- **Pointwise surface containment:** 48.1% with spread_AS, **84.4%** with spread_final.

- **PCA on ΔIV (Appendix):** PC1=87.6% (surface level shift), PC2=7.1%, PC3=3.0%; 3 PCs explain 97.7%.

- **HMM regimes (Appendix):** 3 regimes — Low-vol 23%, Mid-vol 23%, High-vol 54%. High-vol is the dominant regime over 2010–2020.

---

## Supplementary Notebooks

### `S1_feature_engineering_pipeline.ipynb`
Full pipeline for HAR-RV components, SSVI surface features (ATM, slope, skew, curvature), regime indicators (λ_stress), near-arbitrage condition scores (max_cond1), and macro features (VIX, Treasury spread).

### `S2_frontier_analyses.ipynb`
Advanced structural analyses:
- Structural breaks (Chow test): Volmageddon (Feb 2018) and COVID (Mar 2020)
- CUSUM parameter stability
- Granger causality between SSVI parameters and macro variables
- Path-dependence analysis (Andres et al. 2025 framework)

---

## References

- Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance*, 14(1), 59–71.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *JFEC*, 7(2), 174–196.
- Andersen, T., Bollerslev, T., Diebold, F. & Labys, P. (2003). Modeling and forecasting realized volatility. *Econometrica*, 71(2), 579–625.
- Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian VAR models. *Econometrica*, 59(6), 1551–1580.
- Engle, R. & Granger, C. (1987). Co-integration and error correction. *Econometrica*, 55(2), 251–276.
- Diebold, F. & Mariano, R. (1995). Comparing predictive accuracy. *JBES*, 13(3), 253–263.
- Andrès, H., Boumezoued, A. & Jourdain, B. (2025). The implied volatility surface (also) is path-dependent. arXiv.
- Heston, S. (1993). A closed-form solution for options with stochastic volatility. *RFS*, 6(2), 327–343.
