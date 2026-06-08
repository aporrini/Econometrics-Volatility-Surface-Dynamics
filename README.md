# SSVI Volatility Surface Dynamics — S&P 500 Options (2010–2020)

**Politecnico di Milano — Econometrics Project (A.Y. 2025/26)**  
Authors: Alessio Porrini, Marco Amarilli, Camilla Introzzi, Christian Frigerio

---

## Overview

Full analytical pipeline for SSVI (Surface Stochastic Volatility Inspired) implied volatility dynamics and realized volatility forecasting using **SPX daily European options**, with calibrated parameters spanning 2010–2020 (~2,633 trading days).

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
│   ├── A_implied_volatility_panel.ipynb   # Panel IV regression (~1.5M obs)
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
Computes Black-76 IV via vectorized Newton-Raphson. Extracts implied forward prices via put-call parity. Output: ~1.5M IV observations.

### `02_ssvi_calibration.ipynb`
Calibrates SSVI surface per trading day subject to butterfly and calendar no-arbitrage constraints. Writes `Data/ssvi_all_dates_clean_results.csv`. Success rate: **95.1%** (2,633 / 2,768 days), median RMSE_iv < 0.01, zero calendar arbitrage violations.

---

## Primary Analysis Notebooks

### Notebook A — Panel Implied Volatility Regression
`econometric_analysis/A_implied_volatility_panel.ipynb`

Regresses Black-76 IV on option structural features (forward moneyness, TTE, liquidity) across ~1.49 million observations. Uses within-transformation (time-demeaning) to avoid an explicit ~31 GB dummy matrix.

Sections: Data loading → EDA → Pooled OLS → Day FE → Day + Maturity FE → Hausman test (FE vs RE) → RESET test → log(IV) robustness → Gauss-Markov diagnostics → Key findings.

---

### Notebook B — SSVI Parameter Dynamics
`econometric_analysis/B_ssvi_parameter_dynamics.ipynb`

**Data:** 2,632 obs (2010–2025). Train: 2,105 obs (through 2022-06-17). Test: 527 obs (2022–2025).

**Sections:**

| Section | Content |
|---------|---------|
| 1. Data Loading | SSVI params (GitHub), VIX/Treasury rates (FRED) |
| 2. ARMA Modelling | BIC-optimal ARMA for 10 series (5 levels + 5 first differences) |
| 3. ARMAX Modelling | Exogenous: Δlog(VIX), VIX z-score |
| 3b. HAR Modelling | HAR(1,5,22) long-memory forecast of the *differenced* SSVI parameters (Corsi 2009 / Andrès et al. 2025) |
| 4. VAR Analysis | VAR(2, BIC-optimal) on 5-parameter system |
| 5. Cointegration | Johansen + Engle-Granger for α–η |
| 6. PCA on SSVI Surface Changes | HAR on PC scores of Δω — null result (see below) |
| 7. Comprehensive Results | Full horse-race commentary |
| 8. Vol Surface Prediction | Sticky vs ARMA vs VAR surface RMSE |
| Appendix A | Stationarity (ADF + KPSS, 10 series) |
| Appendix B | AR-GARCH prediction intervals |
| Appendix C | Rolling Johansen — regime-conditional cointegration |

**Key actual results:**

- **Stationarity:** All 5 level parameters show conflicting ADF (I(0)) vs KPSS (non-stationary) → classified "Uncertain". First differences are clean I(0). Full parameter descriptives: α mean=−3.45, β mean=1.19, ρ mean=−0.77, η mean=0.75, γ mean=0.54.

- **ARMA OOS (all series):** MSE-ratio = **1.0000**, R²_OOS = **0.0000** across all 10 series. No ARMA specification beats the random walk. BIC-selected orders: α→ARMA(1,0), β/ρ/η/γ→ARMA(1,1).

- **ARMAX OOS:** Same result — MSE-ratio = 1.0000 for all. Despite Granger significance of Δlog(VIX) for α and β (p≈0.017–0.028), the exogenous signal does not translate to OOS improvement.

- **HAR OOS (Section 3b):** a HAR(1,5,22) model on the *differenced* parameters achieves **R²_OOS = 0.51–0.67** vs. the random-walk baseline (MSE-ratio 0.33–0.49) for all five series — beating both the random walk and ARMA. The differenced parameters have *negative* lag-1 autocorrelation (−0.09 to −0.36, bid-ask-bounce-like mean reversion), so a second comparison — HAR against the **expanding historical mean** (a trivial constant forecast) — pins down how much of that edge is genuine multi-horizon structure rather than simple shrinkage toward the mean:

  | Series | HAR R²_OOS (vs RW) | Constant-mean R²_OOS (vs RW) | HAR R²_OOS (vs constant-mean) | Beats constant-mean? |
  |--------|--------------------|------------------------------|-------------------------------|----------------------|
  | d_alpha | 0.5664 | 0.5709 | **−0.0105** | ✗ |
  | d_beta | 0.5506 | 0.5534 | **−0.0064** | ✗ |
  | d_rho | 0.5092 | 0.5331 | **−0.0511** | ✗ |
  | d_eta | 0.5642 | 0.5700 | **−0.0135** | ✗ |
  | d_gamma | 0.6661 | 0.6223 | **+0.1162** | ✓ |

  The constant-mean forecast alone already reaches R²_OOS = 0.53–0.57 vs. random walk — essentially matching HAR for α, β, ρ, η, where HAR adds **nothing** on top of it (R²_OOS vs constant-mean is slightly *negative*). **γ is the sole exception**: HAR genuinely beats the constant-mean forecast (+0.12), consistent with γ's stronger persistence and structural-break profile (S2 Chow tests, rolling-Johansen). So the Andrès et al. (2025)-inspired hypothesis — that multi-horizon history of the parameters predicts their own future innovations — **survives only for γ**; for α, β, ρ, η, daily innovations are essentially unpredictable beyond their own mean, which is exactly what an efficiently-priced, near-martingale surface (Section 7) would predict.

- **VAR(2) OOS:** Mostly MSE > 1. Only γ achieves MSE-ratio = **0.938** (beats RW slightly).

- **Cointegration (α, η):** full-sample Johansen rank = 2 — *full rank* in a bivariate system, i.e. evidence that both series are individually (near-)stationary rather than jointly tied by a single I(1) long-run attractor (which would instead show up as rank = 1). Engle-Granger still finds a significant long-run linear relationship (p = **0.0002***), but VECM OOS performance is essentially nil: α R²=+0.004 (barely positive), η R²=−0.018 (negative → VECM misspecified for η).

- **Rolling Johansen (500-day windows):** rank = 0 in 2.8%, rank = 1 (genuine single cointegrating vector) in **30.0%**, and rank = 2 (full rank — both series individually stationary) in **67.2%** of windows. The estimated rank — and hence the I(1)/I(0) reading of the system — is itself regime-dependent and correlated with the VIX level (EG calm p=0.0004, EG stress p=0.155).

- **PCA on SSVI surface changes (Δω):** 3 PCs explain **99.1%** of variance (PC1=92.69% common daily shock, PC2=4.38% term-structure tilt, PC3=2.04% smile-curvature shock). HAR on these PC scores returns *negative* R²_OOS = **−0.004 / −0.020 / −0.008** at h=1/5/20 — i.e. it underperforms the trivial zero-change (random-walk) baseline at every horizon. This is a clean null result: daily *changes* along the leading principal directions of the surface are near-white-noise, mirroring the near-random-walk behaviour already documented for the individual SSVI parameters (see *Notes on findings* below).

- **Caching:** Expensive OOS loops cached in `output/cache/`. Set `FORCE_RECOMPUTE = True` to regenerate.

---

### Notebook C — Realized Volatility Forecasting (Primary Deliverable)
`econometric_analysis/C_realized_volatility_forecasting.ipynb`

**Data:** 2,671 obs (2010-05-25 to 2020-12-31). Train: ~2,136 obs (through 2018-11-14). Test: 534 obs (2018-11-15 to 2020-12-30).

It is well known that HAR augmented with the VIX outperforms plain HAR for S&P 500 realized-volatility forecasting — but the VIX is an SPX-specific input. This notebook asks a more **portable** question: can a single underlying's *own* option surface (its SSVI levels, ATM IV, and nonlinear stress interactions) supply the same regime information, in a form that generalizes to *any* asset with a liquid, tradable option market? The SSVI-augmented models (F3_ATM, M4_smooth, M4+int) are this VIX-free, surface-only alternative.

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
| **HAR** | 3 | **0.466** | **0.846** | — | **0.857** | — |
| HAR+VIX | 4 | 0.488 | 0.832 | −0.58 | 0.864 | +0.52 |
| F3_ATM | 4 | 0.465 | 0.840 | −2.47** | 0.846 | −1.87* |
| HAR_rhoJ | 4 | 0.465 | 0.846 | −0.72 | 0.857 | −0.38 |
| M1 | 4 | 0.466 | 0.842 | −2.51** | 0.849 | −1.93* |
| M4_smooth | 9 | 0.449 | 0.819 | −3.59*** | 0.806 | −2.78*** |
| M4+int | 12 | 0.451 | 0.825 | −2.82*** | 0.817 | −2.13** |

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

A market-making risk engine that **extends the Avellaneda & Stoikov (2008) optimal-quoting framework** from a single ATM-volatility input to the full SSVI implied-volatility surface. The baseline `spread_AS` reservation-price/inventory-risk quote (driven by one scalar volatility) systematically under-covers realised surface moves; the notebook adds an **SSVI-based residual-risk add-on** calibrated from the cross-section of strikes and maturities, producing `spread_final`.

**Data:** 2,633 trading days, 45-point IV grid (9 strikes × 5 maturities). Split: 70/15/15 chronological (no shuffle). surface_move: mean = 0.00583 vol-units, std = 0.00798.

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

- **Jump detection:** BPV rate = 29.7% (too loose); selected Method A (q95+2σ) → 1.4% jump rate. HAR-J vs HAR DM is not significant (p = 0.290) — jumps add no forecasting improvement.

- **c*(T) term structure:** c*(T) = 4.950 − 3.208√T, R² = **0.949**. c* strictly decreasing: short-dated RV forecasts understate tail risk more severely than long-dated. The √T scaling is consistent with Brownian surface dynamics.

- **Coverage:**

| Bucket | Coverage (spread_AS only) | Coverage (spread_final) |
|--------|--------------------------|------------------------|
| Global | 57.3% | **95.9%** |
| 1M | 19.4% | 97.4% |
| 3M | 48.1% | 96.9% |

The AS-style baseline alone dramatically under-covers (19–57%). The residual-risk add-on (74% of total spread globally) is necessary to reach target coverage.

- **Pointwise surface containment:** 47.5% with spread_AS, **82.9%** with spread_final.

- **PCA on ΔIV (Appendix):** PC1=84.0% (surface level shift), PC2=10.1%, PC3=3.5%; 3 PCs explain 97.5%.

- **HMM regimes (Appendix):** 3 regimes — Low-vol 24% (641 days), Mid-vol 25% (666 days), High-vol 50% (1,316 days). High-vol is the dominant regime over 2010–2020.

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

## Notes on the Review (corrections worth flagging)

A line-by-line review of every executed cell against the markdown narrative surfaced a handful
of genuinely interesting points — some are corrections to stale text, one is a substantive
finding about the data pipeline itself:

1. **The PCA result reverses sign once read correctly.** The pre-review narrative claimed
   Section 6 of Notebook B found *positive* OOS predictability for the surface level
   (R²_OOS = 0.926/0.741/0.493, attributed to a "PC1 = level, mean-reverting" story). The
   notebook actually runs PCA on **surface *changes* Δω**, not levels — and the printed output
   shows *negative* R²_OOS = −0.004/−0.020/−0.008 at every horizon. This is not a minor
   numerical slip: it is the opposite empirical conclusion (a clean null result rather than a
   "key positive finding"), and it is actually the *more* interesting result, because it shows
   the near-random-walk behaviour documented for individual SSVI parameters extends cleanly to
   the dominant joint directions of the surface itself — i.e. the IV surface looks efficiently
   priced both parameter-by-parameter *and* in its principal directions of daily variation.

2. **Johansen rank = 2 does not mean "cointegrated."** The original commentary read the
   full-sample Johansen rank of 2 (for the bivariate {α, η} system) as confirming cointegration.
   In a two-variable VAR, full rank (= 2) means *both series are individually stationary* —
   the genuine single-cointegrating-vector case is rank = 1. The corrected reading (both series
   ≈ I(0), with the relationship itself regime-dependent — rank fluctuates between 0/1/2 across
   rolling windows and tracks the VIX level) is arguably the more interesting econometric story,
   since it reframes the α–η relationship as a structural, regime-conditional feature of the
   SSVI parametrisation rather than a stable tradeable spread.

3. **The γ-sensitivity "regime boundary" was mis-located by one step.** Notebook D's
   Section 7 interpretation placed the sharp collapse of `c*` "between γ = 0.5 and γ = 1.0".
   The printed table shows `c*` only declines moderately there (≈3.1 → ≈2.2, ~30%); the real
   collapse (≈2.2 → ≈0.07, ~97%) happens between **γ = 1.0 and γ = 2.0**. This matters
   practically: it is the point beyond which a wider baseline AS-style quote alone already
   nears the 95% ES coverage target and the HAR-J residual-risk add-on becomes redundant.

4. **A likely date-axis bug in Notebook B (flagged, not fixed).** Notebook B reconstructs
   calendar dates from the `time_elapsed` index via `pd.bdate_range(...)[time_elapsed]` —
   i.e. it treats `time_elapsed` as a *business-day* index — which yields a stated span of
   **2010–2025** (`SSVI: 2,633 successful calibrations | 2010-05-25 to 2025-05-23`). Notebook D
   and `src/ssvi_mm_risk_engine.py::load_ssvi_results`, by contrast, treat `time_elapsed` as a
   **calendar-day** offset (`_SSVI_START + pd.to_timedelta(time_elapsed, unit='D')`), which
   yields **2010–2020** — consistent with every notebook's own title and with the documented
   ~2,768-trading-day sample. The two conventions cannot both be right; we left Notebook B's
   date-handling code untouched (per the "comments/markdown only" review scope) but **the
   reader should treat Notebook B's absolute calendar-date labels as provisional**. This does
   not affect any of the reported statistics — train/test splits, rolling windows, and
   regression results are all defined on the row index, not on the reconstructed calendar date —
   but the date axis on Notebook B's time-series plots and any "as of [date]" narrative should
   be read with this caveat in mind. The one-line fix, if pursued, would be to replace the
   `pd.bdate_range` indexing with the same `_SSVI_START + pd.to_timedelta(time_elapsed, unit='D')`
   logic used in the risk engine.

---

## References

- Andersen, T., Bollerslev, T., Diebold, F. & Labys, P. (2003). Modeling and forecasting realized volatility. *Econometrica*, 71(2), 579–625.
- Andrès, H., Boumezoued, A. & Jourdain, B. (2025). The implied volatility surface (also) is path-dependent. *arXiv preprint*.
- Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3), 217–224.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
- Diebold, F. & Mariano, R. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Engle, R. & Granger, C. (1987). Co-integration and error correction: representation, estimation, and testing. *Econometrica*, 55(2), 251–276.
- Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance*, 14(1), 59–71.
- Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. *Econometrica*, 59(6), 1551–1580.
