# SSVI Surface Dynamics & Realized Volatility Forecasting
## Evidence from S&P 500 Options (2010–2020)

**Research Report — May 2026**  
Politecnico di Milano — Insurance & Econometrics  
Authors: Alessio Porrini, Marco Amarilli, Camilla Introzzi, Christian Frigerio

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Dataset & Methodology](#2-dataset--methodology)
3. [Results by Notebook](#3-results-by-notebook)
   - [Notebook A — Panel IV Regression](#notebook-a--panel-implied-volatility-regression)
   - [Notebook B — SSVI Parameter Dynamics](#notebook-b--ssvi-parameter-dynamics)
   - [Notebook C — Realized Volatility Forecasting](#notebook-c--realized-volatility-forecasting)
   - [Notebook D — Surface Risk Engine](#notebook-d--ssvi-surface-risk-engine)
4. [Key Findings](#4-key-findings)
5. [Limitations](#5-limitations)
6. [Conclusions](#6-conclusions)
7. [References](#7-references)

---

## 1. Executive Summary

This report documents the full empirical analysis of SSVI implied volatility surface dynamics and realized volatility forecasting using approximately **1.818 million S&P 500 option observations** spanning 2010–2020 (with SSVI calibration extended through 2025).

**Headline results:**

- SSVI calibrates on **96.3%** of trading days with median RMSE_iv < 0.01 and zero calendar arbitrage violations.
- Panel IV regression with Day + Maturity Fixed Effects achieves **R² = 0.831** on 1.81M observations; forward moneyness and TTE explain the dominant share of IV cross-sectional variation.
- **Short-memory models fail, but long-memory (HAR) structure reveals genuine predictability in SSVI parameter changes.** All 10 ARMA/ARMAX specifications on levels and first differences achieve MSE-ratio = 1.0000 out-of-sample — no single-lag linear model beats naïve persistence. Yet a HAR(1,5,22) model on the *differenced* parameters (Section 3b) achieves **R²_OOS = 0.51–0.67** (MSE-ratio 0.33–0.49) for all five series, beating both the random walk and ARMA. SSVI parameter changes are not a pure random walk — they carry the same long-memory structure that Corsi (2009) documented for realized volatility; ARMA simply uses the wrong functional form to extract it.
- **HAR is the best realized volatility forecasting model.** All SSVI-augmented specifications (M4_smooth, M4+int, F3_ATM, M1) are significantly *worse* than HAR in the primary evaluation (DM up to −3.47*** at h=5). The main obstacle is distributional shift caused by COVID-19 (VIX reaching ~83) in the test window.
- **Surface-level is forecastable, surface-change is not.** A HAR model on PCA scores achieves R²_OOS = **0.926** at h=1 (vs naive persistence in surface space), but this is a different target from RV forecasting.
- A market-making spread engine with SSVI-based residual-risk add-on achieves **96.2% global ES₉₅ coverage**, compared to 61.2% for the AS-style baseline alone. The add-on coefficient follows c*(T) = 5.413 − 3.488√T (R² = 0.941).

---

## 2. Dataset & Methodology

### 2.1 Data

**Source:** S&P 500 European options panel (2010-01-01 → 2020-12-31).  
**Implied forward:** extracted via put-call parity.  
**Treasury rates:** 1M, 3M, 6M, 1Y, 2Y, 3Y zero-coupon rates from FRED.

**Quality filters:**

| Filter | Threshold |
|--------|-----------|
| Mid price | > $0.05 |
| TTE | > 7 days |
| Bid-ask spread / mid | ≤ 30% |
| Log-moneyness | ∈ [−0.40, 0.30] |

**Scale:**

| Stage | Count |
|-------|-------|
| Raw rows | ~3,331,302 |
| After filters | ~2,324,008 |
| Black-76 IV retained | ~1,818,064 |
| SSVI calibration success | 2,653 / 2,768 days (96.3%) |

### 2.2 SSVI Parametrisation

$$\omega(k,\theta) = \frac{\theta}{2}\left\{1 + \rho\,\phi(\theta)\,k + \sqrt{(\phi(\theta)\,k+\rho)^2 + (1-\rho^2)}\right\}, \quad \phi(\theta) = \frac{\eta\,\theta^{-\gamma}}{1+\eta\,\theta^{1-\gamma}}$$

| Parameter | Economic role | Mean (2010–2025) |
|-----------|--------------|-----------------|
| α | log(ATM variance) — surface level | −3.45 |
| β | Term-structure slope | 1.19 |
| ρ | Skew / leverage effect | −0.77 |
| η | Vol-of-vol / smile width | 0.75 |
| γ | Term-structure decay rate | 0.54 |

### 2.3 Evaluation Metrics

**R²_OOS** (relative to naïve benchmark):
$$R^2_\text{OOS} = 1 - \frac{\sum_t (y_t - \hat{y}_t)^2}{\sum_t (y_t - y_{t-1})^2}$$

**DM-HLN test:** Diebold-Mariano with Harvey-Leybourne-Newbold small-sample correction and HAC Newey-West standard errors. Sign convention: **DM > 0 = model beats HAR, DM < 0 = model worse than HAR**.

---

## 3. Results by Notebook

### Notebook A — Panel Implied Volatility Regression

`econometric_analysis/A_implied_volatility_panel.ipynb`

**Data:** 1,810,000+ observations over ~2,780 trading days.

**Setup:** Regress Black-76 IV on forward moneyness (k), TTE, and bid-ask liquidity measure. Main specifications: Pooled OLS → Day FE → Day + Maturity FE. Within-transformation avoids an explicit 37,500-column dummy matrix.

**Key results:**

| Specification | R² |
|--------------|----|
| Pooled OLS | ~0.65 |
| Day FE | ~0.79 |
| Day + Maturity FE | **0.831** |

- Forward moneyness and TTE are the dominant structural drivers of IV cross-section.
- Hausman test rejects random effects (p < 0.001) — fixed effects are appropriate.
- RESET test indicates mild nonlinearity; log(IV) specification is more linear.
- Gauss-Markov diagnostics: heteroskedasticity present; HAC standard errors used throughout.

---

### Notebook B — SSVI Parameter Dynamics

`econometric_analysis/B_ssvi_parameter_dynamics.ipynb`

**Data:** 2,652 obs (2010-05-25 to 2025-05-23). Train: 2,121 obs (through 2022-06-08). Test: 531 obs.

#### Stationarity (Appendix A)

All 5 level parameters show conflicting ADF (I(0)) vs KPSS (non-stationary) → classified "Uncertain". First differences are clean I(0) for all series.

#### ARMA Modelling

BIC-optimal orders: α→ARMA(1,0), β/ρ/η/γ→ARMA(1,1).

**OOS results (all 10 series — 5 levels + 5 differences):**

| Metric | All series |
|--------|-----------|
| MSE-ratio | **1.0000** |
| R²_OOS | **0.0000** |

No ARMA specification beats the naïve random walk. SSVI parameters exhibit near-unit-root behaviour with no exploitable autocorrelation structure.

#### ARMAX Modelling

Exogenous inputs: Δlog(VIX), VIX z-score. Despite Granger significance of Δlog(VIX) for α and β (p ≈ 0.017–0.028) in-sample, the exogenous signal does **not** translate to OOS improvement. MSE-ratio = 1.0000 for all ARMAX specifications.

#### HAR Modelling — Long-Memory Forecasting of the Differenced Parameters (Section 3b)

Motivated by Corsi (2009) and the path-dependence evidence of Andrès, Boumezoued & Jourdain (2025), a HAR(1,5,22) regression — the same daily/weekly/monthly long-memory averaging structure used for realized volatility — is fitted to the **first-differenced** SSVI parameter series (the stationary representation established in Appendix A) and evaluated out-of-sample with the same rolling-origin protocol as the ARMA/ARMAX models:

| Series | HAR MSE-ratio | HAR R²_OOS | ARMA MSE-ratio | Beats RW | Beats ARMA |
|--------|---------------|------------|----------------|----------|------------|
| d_alpha | 0.4336 | **0.5664** | 1.0000 | ✓ | ✓ |
| d_beta | 0.4494 | **0.5506** | 1.0000 | ✓ | ✓ |
| d_rho | 0.4908 | **0.5092** | 1.0000 | ✓ | ✓ |
| d_eta | 0.4358 | **0.5642** | 1.0000 | ✓ | ✓ |
| d_gamma | 0.3339 | **0.6661** | 1.0000 | ✓ | ✓ |

**This is the single most important predictability result for the SSVI parameter system.** Where every single-lag ARMA/ARMAX specification is statistically indistinguishable from a random walk (MSE-ratio = 1.0000), the HAR's three-horizon (1/5/22-day) rolling-mean structure recovers **50–67% of the variance reduction relative to the random walk** for every one of the five differenced parameters — with the wing-decay parameter γ showing the strongest gain (R²_OOS = 0.666). The contrast is the key insight: the parameters' first differences are *not* devoid of structure — they are simply not capturable by a low-order autoregressive form. They instead respond to the same multi-horizon, long-memory aggregation that governs realized-volatility dynamics, which is exactly the mechanism Corsi (2009) formalised and Andrès et al. (2025) extended to the implied-volatility surface itself.

#### VAR Analysis

VAR(2, BIC-optimal) on the 5-parameter system.

| Series | MSE-ratio |
|--------|----------|
| α | ≥ 1 |
| β | ≥ 1 |
| ρ | ≥ 1 |
| η | ≥ 1 |
| **γ** | **0.938** |

Only γ achieves a marginal OOS gain under VAR. The system as a whole does not benefit from VAR cross-dynamics.

#### Cointegration

Johansen test (α, η): rank = 2; Engle-Granger p = **0.0002***. The pair is statistically cointegrated over the full sample. VECM OOS: α R² = +0.004 (barely positive), η R² = −0.018 (negative — VECM is misspecified for η OOS).

Rolling Johansen (500-day windows): rank = 1 (genuine single cointegrating vector) in only **29.7%** of windows; rank = 2 in 63.6%. Regime-conditional: EG calm p = 0.0004, EG stress p = 0.048. Cointegration is present but unstable across regimes.

#### PCA on SSVI Surface (Section 6)

| PC | Variance explained |
|----|-----------------|
| PC1 (Level) | 96.34% |
| PC2 (Slope) | 2.76% |
| Total (2 PCs) | **99.1%** |

HAR on PC scores:

| Horizon | R²_OOS (vs naive persistence) |
|---------|-------------------------------|
| h=1 | **0.926** |
| h=5 | **0.741** |
| h=20 | **0.493** |

Surface *levels* are highly persistent and predictable. Note: this is a different forecasting target from daily RV — it measures how well we forecast the next-day surface level rather than realized volatility.

#### Vol Surface Prediction (Section 8)

Horse-race between sticky-delta, ARMA-based, and VAR-based surface reconstruction:
- ARMA and VAR surface RMSE broadly comparable.
- Sticky-delta is the toughest naive benchmark for short horizons.

---

### Notebook C — Realized Volatility Forecasting

`econometric_analysis/C_realized_volatility_forecasting.ipynb`

**Data:** 2,671 obs (2010-05-25 to 2020-12-31). Train: ~2,136 obs (through 2018-11-14). Test: 534 obs (2018-11-15 to 2020-12-30, includes COVID-19 episode).

**Target:** $y_t^{(h)} = \frac{1}{h}\sum_{i=1}^h \log\text{RV}_{t+i}$ for $h \in \{1, 5, 20\}$.

#### VIF / Feature Selection (Section 6a)

Uncentered lambda interactions: VIF = **3,738** (lambda_stress). Centering $\tilde\lambda_t = \lambda_t - \bar\lambda_\text{train}$ resolves the collinearity. Feature selection by OLS significance at h=5 (training fold only) retains {α, ρ, γ} as SSVI levels and {λ̃, α·λ̃, ρ·λ̃} as interactions → M4_smooth (9 features).

At h=20, M4_smooth OLS shows only γ is significant (p = 0.006); α, ρ, λ̃, and interactions are not significant — consistent with overparameterisation.

#### Single-Split OOS Results (primary evaluation)

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

**HAR is the best model.** All SSVI-augmented specifications are significantly *worse* than HAR. DM convention: positive = model beats HAR, negative = model is worse than HAR.

#### Expanding Window OOS (2016–2020)

| Model | h=5 R²_OOS | DM(h=5) | h=20 R²_OOS | DM(h=20) |
|-------|-----------|---------|-------------|---------|
| **HAR** | **0.853** | — | **0.894** | — |
| HAR+VIX | 0.865 | +1.13 | 0.901 | +1.06 |
| F3_ATM | 0.853 | −0.35 | 0.894 | −0.38 |
| M4_smooth | 0.850 | −0.89 | 0.889 | −0.93 |
| M4+int | 0.853 | +0.06 | 0.897 | +0.76 |

In the expanding window, HAR+VIX is marginally best but DM is not significant (p ≈ 0.26). All SSVI models converge to HAR performance. HAR remains the correct benchmark and winner.

#### Interpretation

The primary obstacle to SSVI augmentation is the COVID-19 distributional shift (VIX ~83 in March 2020). SSVI regime indicators trained on 2010–2018 data cannot generalise to this out-of-distribution event. HAR's parsimony and backward-looking structure provides robustness that richer models lack.

The structural collinearity of SSVI features (VIF analysis) and the overparameterisation of M4_smooth at long horizons further confirm that HAR is the parsimonious optimal model for this dataset.

#### Appendix A — Path-Dependence Test

Following Andres et al. (2025), lagged levels of ρ and γ were added to HAR. Neither achieves meaningful OOS improvement — path-dependence in the SSVI surface does not translate to predictable RV dynamics in this dataset.

---

### Notebook D — SSVI Surface Risk Engine

`econometric_analysis/D_ssvi_surface_risk_addon.ipynb`

**Data:** 2,652 trading days; 45-point IV grid (9 strikes × 5 maturities). Split: 70/15/15 chronological. surface_move: mean = 0.00536, std = 0.00789 vol-units.

#### Jump Detection

| Method | Jump rate |
|--------|----------|
| BPV | 31.3% (too loose) |
| Method A (q95 + 2σ) | **1.4%** (selected) |

HAR-J vs HAR DM: p = 0.959 — jumps add no forecasting improvement.

#### c*(T) Term Structure

$$c^*(T) = 5.413 - 3.488\sqrt{T}, \quad R^2 = 0.941$$

c* is strictly decreasing in T: short-dated IV forecasts understate tail risk more severely than long-dated ones. The √T scaling is consistent with Brownian surface dynamics.

#### Coverage Evaluation

| Bucket | spread_AS only | spread_final |
|--------|---------------|-------------|
| Global | 61.2% | **96.2%** |
| 1M | 28.4% | 97.7% |
| 3M | 54.8% | 97.5% |

The residual-risk add-on accounts for **74% of total spread** globally. The AS-style baseline alone is insufficient; the add-on is necessary to reach target ES₉₅ coverage.

**Pointwise surface containment:** 48.1% (spread_AS), **84.4%** (spread_final).

#### Appendix — PCA on ΔIV & HMM Regimes

| Component | Variance |
|-----------|---------|
| PC1 (surface level shift) | 87.6% |
| PC2 | 7.1% |
| PC3 | 3.0% |
| Total (3 PCs) | 97.7% |

**HMM regimes (3 states):**
- Low-vol: 23% of days
- Mid-vol: 23% of days
- High-vol: **54% of days** — dominant regime over 2010–2020

---

## 4. Key Findings

### Finding 1: Short-memory models fail — but long-memory (HAR) structure uncovers genuine predictability in SSVI parameter changes

All 10 ARMA and ARMAX specifications on the SSVI parameter levels and first differences achieve MSE-ratio = 1.0000 out-of-sample — no single-lag linear model beats the random walk, and the in-sample Granger significance of Δlog(VIX) for α and β (p ≈ 0.02) does not survive out-of-sample. Taken alone, this would suggest the SSVI parameter process is a near-random-walk, consistent with the Efficient Market Hypothesis applied to the volatility surface.

However, a HAR(1,5,22) model fitted to the **differenced** parameters tells a materially different story (Section 3b): it achieves **R²_OOS = 0.51–0.67** (MSE-ratio 0.33–0.49) for all five series, beating both the random walk and the matching ARMA specification by a wide margin. The resolution is that SSVI parameter *changes* are not devoid of structure — they are simply invisible to a low-order autoregressive lens. They respond instead to the same multi-horizon, long-memory averaging mechanism that Corsi (2009) formalised for realized volatility, and that Andrès, Boumezoued & Jourdain (2025) show also governs the path-dependence of the implied-volatility surface. The corrected reading is therefore: **the SSVI surface is not an efficient random walk at the single-lag level, but its short-run dynamics are governed by long-memory, multi-horizon structure that only a HAR-type model can extract.**

### Finding 2: HAR dominates all SSVI-augmented RV models

In the primary single-split evaluation, HAR at h=5 achieves R²_OOS = 0.846. Every SSVI-augmented model is significantly worse (DM: F3_ATM −2.48**, M1 −2.51**, M4_smooth −3.47***). The richer SSVI models overfit the training distribution and fail to generalise through COVID-19. This is a *meaningful* negative result: SSVI surface features contain no incremental predictive power for RV beyond what HAR already captures.

### Finding 3: Surface levels are highly persistent and forecastable

PCA on the SSVI surface: 2 PCs explain 99.1% of variance. HAR on PC1/PC2 achieves R²_OOS = 0.926 at h=1. This is a forecasting result about the *level* of the implied variance surface, not about RV. The contrast between this high R² and the failure of SSVI augmentations in notebook C confirms that knowing the surface level does not help forecast future realised volatility beyond the HAR benchmark.

### Finding 4: Cointegration between α and η is real but regime-conditional

Full-sample Engle-Granger p = 0.0002***. However, rolling Johansen finds rank = 1 in only 29.7% of windows. VECM OOS performance is near zero or negative. The α–η relationship is a structural feature of SSVI's parametrisation rather than an actionable predictive signal.

### Finding 5: Residual-risk add-on is necessary for practical coverage

The √T term structure of c*(T) is well-calibrated (R² = 0.941). Without the add-on, global coverage is only 61.2%; with it, 96.2%. The maturity structure of the add-on is economically meaningful: short-dated IV moves are more volatile relative to their long-run expectation, requiring larger risk buffers.

---

## 5. Limitations

1. **Distributional shift.** The 2018–2020 test window includes COVID-19, which is an extreme OOS regime. The negative SSVI results may not generalise to calmer test periods (2015–2018 would be a useful robustness check).

2. **Single underlying.** Results are specific to SPX options. Different underlyings (individual equities, FX) may show different SSVI predictability.

3. **SSVI specification.** The power-law φ(θ) specification is one of several SSVI variants. Results may differ under raw SVI, eSSVI, or SABR parameterisations.

4. **VECM instability.** Rolling Johansen shows regime-conditional cointegration. The VECM's failure OOS may reflect structural breaks (Volmageddon Feb 2018, COVID Mar 2020) rather than a fundamental lack of cointegration.

5. **Expanding window convergence.** In the expanding window evaluation, SSVI models converge to HAR performance, suggesting the single-split deficit is partly due to insufficient training data for SSVI features in the early window. A longer time series might alter conclusions.

---

## 6. Conclusions

The main empirical result of this project is essentially **negative with informational content**: the rich SSVI surface structure does not improve upon the simple HAR model for realized volatility forecasting. This is not a failure of the analysis — it is a meaningful finding about market efficiency and the limits of structural IV features for short-horizon RV prediction.

The project delivers three distinct positive contributions:

1. **A fully arbitrage-free SSVI calibration database** (2,652 daily surfaces, 96.3% success rate) that can serve as a foundation for future research.
2. **A panel IV decomposition** showing that 83.1% of IV cross-sectional variation is explained by moneyness, TTE, and fixed effects alone — SSVI provides the *shape*, not the predictive signal.
3. **A practical surface risk engine** (notebook D) with a calibrated √T residual-risk add-on achieving 96.2% ES₉₅ coverage — the most direct operational contribution.

The PCA-HAR result (R²_OOS = 0.926 at h=1 for surface level) is a positive side finding, demonstrating that the implied variance surface is jointly forecastable as a low-dimensional system even when individual SSVI parameters are not.

---

## 7. References

- Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance*, 14(1), 59–71.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *JFEC*, 7(2), 174–196.
- Andersen, T., Bollerslev, T., Diebold, F. & Labys, P. (2003). Modeling and forecasting realized volatility. *Econometrica*, 71(2), 579–625.
- Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian VAR models. *Econometrica*, 59(6), 1551–1580.
- Engle, R. & Granger, C. (1987). Co-integration and error correction. *Econometrica*, 55(2), 251–276.
- Diebold, F. & Mariano, R. (1995). Comparing predictive accuracy. *JBES*, 13(3), 253–263.
- Andrès, H., Boumezoued, A. & Jourdain, B. (2025). The implied volatility surface (also) is path-dependent. arXiv.
- Heston, S. (1993). A closed-form solution for options with stochastic volatility. *RFS*, 6(2), 327–343.
- Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.
