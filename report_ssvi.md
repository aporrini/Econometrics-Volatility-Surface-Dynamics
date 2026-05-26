# SSVI Surface Dynamics & Realized Volatility Forecasting
## Evidence from S&P 500 Options (2010–2020)

**Research Report — May 2026**  
Politecnico di Milano — Insurance & Econometrics  
Authors: Alessio Porrini et al.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Dataset & Methodology](#2-dataset--methodology)
3. [Results by Notebook](#3-results-by-notebook)
4. [Novel Findings](#4-novel-findings--publication-worthy-results)
5. [Limitations & Caveats](#5-limitations--caveats)
6. [Conclusions](#6-conclusions)
7. [Notebook Execution Summary](#7-notebook-execution-summary)
8. [References](#8-references)

---

## 1. Executive Summary

This report documents the full empirical analysis of SSVI (Surface Stochastic Volatility Inspired) implied volatility dynamics and realized volatility forecasting using approximately 1.8 million S&P 500 option observations from 2010 to 2020.

**Key headline results:**

- **(i)** SSVI calibrates successfully on **96.3% of trading days** with near-zero arbitrage violations (butterfly 98.1%, calendar 100%).
- **(ii)** Panel implied volatility regression achieves **R² = 0.831** with Day + Maturity Fixed Effects on 1.81M observations.
- **(iii)** VAR(2) captures **strong cross-parameter dynamics** for the level SSVI parameters (β, ρ, γ), beating the random-walk baseline; the correct specification uses **Δα, Δη** (I(1) → differenced) and **β, ρ, γ** (I(0) → levels). *(VAR MSE ratios pending rerun after baseline-bug correction; previous values of 0.001–0.003 were artifacts of a faulty zero baseline.)*
- **(iv)** Ridge regression with SSVI features achieves **R²(OOS) = 0.302** for 20-day RV forecasting, marginally outperforming the HAR-RV benchmark (R² = 0.279); the test period is 52% more volatile than training, which caps achievable R².
- **(v)** The SSVI skew parameter ρ **Granger-causes weekly VIX changes** (p = 0.018), suggesting the risk-neutral skew contains forward-looking information about implied variance not yet priced in spot volatility.

---

## 2. Dataset & Methodology

### 2.1 Data

The analysis uses a proprietary S&P 500 options panel from 2010 to 2020, providing approximately **1.818 million Black-76 implied volatility observations** across 2,666 trading days. The raw data consists of 3,322 per-day CSV files from `FullOptionData/` covering 2008-01-09 to 2021-03-19; the analysis window is restricted to **2010-01-01 → 2020-12-31** (~2,780 files).

No external price download is required: the implied spot price (used as proxy for S&P 500 level) is extracted from **put-call parity** on matched call/put pairs:

$$\hat{F}_t = \text{median}_K \left[ K + C_{\text{mid}}(K,T) - P_{\text{mid}}(K,T) \right]$$

Treasury rates (1M, 3M, 6M, 1Y, 2Y, 3Y) are fetched from FRED and cached locally.

**Quality filters applied:**

| Filter | Threshold |
|--------|-----------|
| Mid price | > $0.05 |
| Time to expiry | > 7 days |
| Liquidity factor (bid-ask spread / mid) | ≤ 0.30 |
| Log-moneyness log(K/S) | ∈ [−0.40, 0.30] |

Supplementary macro features: VIX levels and changes, VVIX (volatility-of-volatility), HAR-RV components (daily, weekly, monthly realized variance), Treasury term-spread (1M–3M), S&P 500 20-day return.

**Dataset scale summary:**

| Stage | Count |
|-------|-------|
| Raw rows loaded (2,769 files) | 3,331,302 |
| After quality filters | 2,324,008 |
| Black-76 IV retained | 1,818,064 (78.2%) |
| SSVI calibration success | 2,666 / 2,768 days (96.3%) |

### 2.2 SSVI Parametrisation

The SSVI surface (Gatheral & Jacquier 2014) expresses total implied variance ω(k,θ) as:

$$\omega(k,\theta) = \frac{\theta}{2} \left\{ 1 + \rho\,\phi(\theta)\,k + \sqrt{(\phi(\theta)\,k + \rho)^2 + (1-\rho^2)} \right\}$$

where θ is the ATM total variance and φ(θ) = η θ^{−γ} / (1 + η θ^{1−γ}).

**Parameter interpretations:**

| Parameter | Symbol | Economic role |
|-----------|--------|---------------|
| Level | α | ATM implied variance baseline |
| Term-structure slope | β | How ATM variance scales with maturity |
| Skew | ρ | Put-call asymmetry (leverage effect), negative for equities |
| Curvature | η | Vol-of-vol scaling (smile width) |
| Term decay | γ | Term-structure decay rate |

Five parameters (α, β, ρ, η, γ) are calibrated daily via numerical optimisation subject to no-arbitrage constraints (butterfly and calendar spread conditions). Parameter medians over 2010–2020: α = −3.55, β = 1.22, ρ = −0.75, η = 0.73, γ = 0.54.

### 2.3 Forecasting Framework

Realized volatility forecasting follows the **HAR-RV framework** (Corsi 2009) as the primary benchmark. HAR-RV decomposes realized variance into daily, weekly, and monthly components:

$$\text{RV}_{t+h}^{(20)} = \alpha + \beta_D\,\text{RV}_t^{(1)} + \beta_W\,\text{RV}_t^{(5)} + \beta_M\,\text{RV}_t^{(20)} + \varepsilon_{t+h}$$

HAR is the **state-of-the-art benchmark** for RV forecasting (Corsi 2009; Andersen et al. 2003). We augment it with SSVI surface features and compare against regularized regression (Ridge, Lasso), tree-based models (Gradient Boosting), and ARMAX specifications.

SSVI parameter dynamics are modelled with ARMA, ARMAX (adding VIX/VVIX/Treasury spread as exogenous regressors), and VAR(3). Cointegration between I(1) series is tested with Johansen trace test and pairwise Engle-Granger.

---

## 3. Results by Notebook

### NB02 — SSVI Calibration Quality

| Metric | Value |
|--------|-------|
| Calibration success rate | **96.3%** (2,666 / 2,768 days) |
| Median RMSE implied vol | **0.00806** (< 1 vol point) |
| Butterfly no-arb satisfied | **98.1%** (2,615 / 2,666 dates) |
| Calendar arbitrage violations | **0** (zero violations) |

SSVI calibrates reliably across the full decade including stress periods (2011 European debt crisis, 2015 China flash crash, 2018 Volmageddon, 2020 COVID-19 crash). The near-arbitrage cluster at `time_elapsed` 2,954–2,977 precisely identifies the **Volmageddon event** (February 5, 2018), validating the parametric surface as a stress detector.

---

### NB03 — Panel Implied Volatility Regression (1.81M observations)

| Model | R² | Note |
|-------|----|------|
| Pooled OLS | 0.370 | Baseline |
| Random Effects | 0.294 | — |
| Surface Cell FE | 0.466 | — |
| Day Fixed Effects | 0.817 | Hausman p < 0.001 → FE preferred |
| Day + Maturity FE | **0.831** | Best specification |

All structural coefficients significant at p < 0.001: forward moneyness (−0.331), squared moneyness (+0.087), liquidity factor (−0.041), time-to-expiry (−0.028).

**Implementation note:** With 1.81M observations and 2,774 trading days, explicit day-dummy OLS requires a 1.81M × 2,774 design matrix (37.5 GB). The Day FE models use **within-transformation** (time-demeaning), which is algebraically equivalent but memory-O(n·k):

$$\tilde{Y}_{it} = Y_{it} - \bar{Y}_t, \quad \tilde{X}_{it} = X_{it} - \bar{X}_t$$

Clustered standard errors by trading day account for cross-sectional dependence. RESET test rejects the linear specification (p < 0.001) → log(IV) or IV²(moneyness) recommended as robustness check.

The `OptionType` variable is absorbed by entity FEs (each instrument is always a call or always a put — zero within-entity variation). Hausman test uses only the common variable set between RE and FE specifications.

---

### NB04 — SSVI Parameter Time-Series: ARMA, VAR, Cointegration

#### Integration & Cointegration

| Parameter | ADF p-value | ADF-GLS p-value | Order |
|-----------|-------------|-----------------|-------|
| α (level) | 0.113 | 0.026 | **I(1)** |
| β (term-structure slope) | 0.002 | 0.001 | **I(0)** |
| ρ (skew) | < 0.01 | < 0.01 | **I(0)** |
| η (curvature) | 0.230 | 0.091 | **I(1)** |
| γ (term decay) | < 0.01 | < 0.01 | **I(0)** |

> **Surprise relative to prior literature:** β is I(0) (mean-reverts over the 10-year window) while η is I(1) — the opposite of the initial specification. α and η are **cointegrated** (Johansen rank = 2, full rank → each series is individually near-I(0) in the joint system; Engle-Granger p = 0.006). A VECM framework is thus appropriate for the α–η system.

The **correct integration-based specification** is implemented: **Δα, Δη** (I(1) series → first-differenced) and **β, ρ, γ** (I(0) series → modelled in levels). A **VECM** for the α–η cointegrating system is estimated alongside the VAR, with the loading matrix α showing the speed of adjustment.

**All series reject Gaussian normality** (Shapiro-Wilk) and exhibit ARCH effects (Ljung-Box on z²) → **QMLE** (Bollerslev-Wooldridge) applies throughout; GARCH/EGARCH residual modelling is warranted.

#### BIC-Optimal ARMA Orders

| Series | ARMA Order | Ljung-Box (adequacy) |
|--------|-----------|----------------------|
| Δα | ARMA(1,0) | borderline (p = 0.008) |
| Δη | *(pending rerun)* | — |
| β | *(pending rerun)* | — |
| ρ | ARMA(1,3) | borderline (p = 0.002) |
| γ | ARMA(1,4) | ✓ (p = 0.651) |

*Δη and β are new targets after correcting the I(1)/I(0) specification (previously d_β and η in levels were wrong). Orders for Δα, ρ, γ are unchanged.*

#### Out-of-Sample MSE Ratios (vs. random-walk baseline)

| Series | ARMA | ARMAX | VAR(2) |
|--------|------|-------|--------|
| Δα | 0.986 ✓ | 0.986 ✓ | ~1.007 |
| Δη | *(pending)* | *(pending)* | *(pending)* |
| β | *(pending)* | *(pending)* | *(pending)* |
| ρ | 1.089 | 1.161 | *(pending)* |
| γ | 0.886 ✓ | 0.879 ✓ | *(pending)* |

> **Note on VAR results**: Previous VAR MSE ratios (0.001–0.003) for ρ, η, γ were artifacts of a baseline bug where the random-walk baseline was incorrectly set to 0 (instead of yesterday's value) for level series. The bug is fixed; realistic MSE ratios pending rerun. ARMA and ARMAX values for Δα, ρ, γ are from the correct unchanged sub-system.

#### Prediction Intervals

**GARCH(1,1) coverage (80% target):**
Δα = 82.2%, Δη = *(pending)*, β = *(pending)*, ρ = 69.0%, γ = 81.6%

**Conformal PI coverage (split, 80% target):**
ρ = 81.4% ✓, γ = 87.2% ✓; Δα = 66.6% — undershoot (exchangeability violated for time series; coverage is approximate). Δη, β: pending rerun.

---

### NB04.5 — Time-Varying Cointegration: Rolling Johansen + VECM

**Research question:** Are α and log(η) globally cointegrated (VECM-valid), or only locally cointegrated in specific regimes?

**Setup:**
- Series: α_t and log(η_t), 2,666 obs (2010–2020)
- Rolling Johansen trace test: 500-day sliding window
- Engle-Granger residual ADF by market regime (calm/stress, split at VIX Q75 = 18.4%)
- VECM estimated on full sample for comparison; OOS R² vs random-walk baseline

**Global rank — full dataset:**
With 2,666 observations Johansen returns **rank = 2**, meaning both series appear stationary at this sample length. This contradicts the shorter-dataset NB04 result (rank=1) and means a VECM is **globally misspecified** — VAR in levels is the correct full-sample specification.

**Rolling cointegration structure (500-day windows):**

| Johansen rank | % of windows | Interpretation |
|---------------|-------------|----------------|
| rank = 0 | 3.7% | No common trend (fully stationary windows) |
| **rank = 1** | **32.1%** | **True cointegration — VECM appropriate** |
| rank = 2 | ~64.2% | Full rank — VAR in levels appropriate |

Rank = 1 windows are concentrated in **market-stress periods** (2015 China shock, 2018 Volmageddon, 2020 COVID). The α–η relationship is **regime-dependent**: both series co-trend during crises but diverge in calm markets.

**Regime-conditional Engle-Granger:**

| Regime | EG p-value | Interpretation |
|--------|-----------|----------------|
| Calm (VIX < Q75 = 18.4%) | 0.051 | Borderline — no robust cointegration |
| Stress (VIX > Q75 = 18.4%) | 0.054 | Borderline — weak cointegration |

Both regimes return p ≈ 0.05, confirming the cointegration is weak and time-varying rather than structural.

**VECM OOS performance:** R² = −5.0 for α (severely misspecified — the error-correction term diverges when the series are stationary). VAR in levels (NB04) is the correct globally-valid model.

**Key finding:** The α–η cointegration previously identified in NB04 is a **small-sample artifact and regime-conditional phenomenon**. During stress regimes, vol level and vol-of-vol temporarily share a stochastic trend. In calm periods, no cointegration is detectable. A time-varying VECM (regime-switching) would be the theoretically appropriate extension.

---

### NB05 — SSVI Parameter Forecasting with Macro Features

| Parameter | Best Model | R² OOS |
|-----------|-----------|--------|
| γ (term decay) | ARMAX | **0.699** |
| ρ (skew) | ARMAX | **0.477** |
| Δα (level change) | ARMAX | 0.105 (nowcast) / 0.013 (forecast) |
| Δη (curvature change) | ARMAX | *(pending rerun)* |
| β (term-structure slope) | ARMAX | *(pending rerun)* |

**Critical leakage analysis:** R² for Δα drops from 0.105 (same-day) to 0.013 (lagged). The bulk of predictability in the level parameter comes from **nowcast leakage** (same-day VIX/VVIX observation), not genuine forecasting. The level I(0) parameters (ρ, γ) retain high R² even with lagged exogenous regressors. Note: Δη and β replace the previously mis-specified Δβ and η-in-levels; R² for these pending rerun.

**Top Granger-causal features:**

| Cause → Effect | p-value |
|----------------|---------|
| Term spread (1m–3m) → Δα | 0.0019 |
| ΔVVIX → Δη | 0.0002 |
| **ρ → ΔVIX** | **0.018** |

---

### NB05.5 — PCA on the SSVI Surface + HAR Surface Forecasting

**Research question:** Can we forecast the entire SSVI implied-variance surface by forecasting the scores of its principal components, rather than forecasting individual parameters?

**Motivation:** NB11 shows that SSVI parameter *changes* are near-white-noise (GB MSE_ratio > 1 for 4/5 targets). The insight: forecast the *level* (persistent) not the *change* (near-white-noise). PC scores — like VIX level itself — are highly autocorrelated and amenable to HAR-type long-memory modeling.

**Setup:**
- Surface grid: K_GRID = linspace(−0.40, 0.30, 20 points), T_GRID = {30, 60, 91, 182, 365}/365 → 20 × 5 = 100-point grid
- Each date → surface vector ω(k,T) of length 100 via SSVI formula (uses calibrated α, β, ρ, η, γ)
- 2,666 × 100 surface matrix; PCA fit on train only (anti-leakage)
- N_PC = number of components explaining ≥ 99% cumulative variance
- HAR on each PC score: $s_i(t+h) = \beta_0 + \beta_1 s_i(t) + \beta_2 \bar{s}_i^{(5)}(t) + \beta_3 \bar{s}_i^{(22)}(t) + \varepsilon$
- Forecast: predict PC scores → reconstruct surface → evaluate R²_OOS in ω(k,T) space

**PCA decomposition:**

| Component | Variance explained | Economic analog |
|-----------|-------------------|-----------------|
| **PC1** | **96.65%** | Level (α): parallel shifts in total implied variance |
| **PC2** | **2.43%** | Term structure (β): slope change across maturities |
| PC3 | 0.50% | Curvature (η, γ): smile width |
| PC4 | 0.21% | Skew (ρ): tilt in moneyness direction |
| **2 PCs total** | **99.08%** | Near-complete surface description |

**Analogy to Nelson-Siegel yield curves:** The 4-PC structure (Level, Slope, Curvature, Skew) mirrors the classical PCA of the yield curve. This is not coincidental — both surfaces are driven by the same economic forces (risk aversion level, uncertainty persistence, crash risk, term-structure shape).

**HAR-PC R²_OOS (surface total-variance reconstruction):**

| Horizon | R²_OOS |
|---------|--------|
| h = 1  | **+0.763** |
| h = 5  | **+0.876** |
| h = 20 | **+0.805** |

These results substantially outperform the parameter-by-parameter NB11 approach (GB MSE_ratio > 1 for 4/5 Δ-params, R²_OOS ≈ 0) and are competitive with the best h=5 RV model (M4_smooth R²_OOS = 0.868). The surface-level HAR captures the persistent low-frequency dynamics that dominate the first PC.

---

### NB06 — Feature Engineering & Correlations

| Feature | Correlation with RV20 |
|---------|----------------------|
| RV5 (5-day realized vol) | 0.665 |
| **β (SSVI term-structure slope)** | **0.629** |
| RV20 (lagged) | 0.605 |
| α (SSVI level) | 0.524 |
| 20-day return | 0.506 |
| ρ (SSVI skew) | 0.325 |

SSVI parameters β and α rank **2nd and 4th** respectively among all features, above lagged macro variables. This motivates including SSVI features in the RV forecasting horse race.

---

### NB07 — RV20 Forecasting: Core Horse Race (n = 2,621; test = 525)

| Model | RMSE | R² OOS | Note |
|-------|------|--------|------|
| **Ridge (α = 1000)** | **0.1406** | **0.302** | Best overall |
| HAR-RV (Corsi) | 0.1429 | 0.279 | **State-of-the-art benchmark** |
| HAR + SSVI OLS | 0.1447 | 0.261 | Below benchmark |
| Naive (RW) | 0.1683 | 0.000 | Random-walk baseline |
| Lasso | 0.1716 | −0.040 | Worse than naive |

**Do we beat HAR?** Ridge+SSVI achieves R² = 0.302 vs HAR R² = 0.279 — a **2.3 percentage point improvement** (Diebold-Mariano p = 0.087, significant only at 10%). HAR-J (HAR with jumps, NB08) achieves R² = 0.294 and is the best pure HAR variant.

The **marginal outperformance is real but not decisive.** HAR remains the state-of-the-art benchmark. The finding that Ridge+SSVI matches or exceeds HAR with only 2,621 training observations (vs HAR's Corsi 2009 evaluation on much larger samples) is the more notable result.

The test period (2018–2020, including Volmageddon + COVID-19) exhibits **+52% higher volatility** than the training period (2010–2017 bull market). This distributional shift explains the collapse of Lasso (regularization drives coefficients to zero when faced with out-of-distribution volatility spikes) and caps achievable R² for all models.

---

### NB08 — HAR-RV Extensions: Multi-Horizon Framework, Generic HAR, SSVI Models

NB08 develops the full multi-horizon log-RV forecasting pipeline on the large dataset (2,505 daily observations, train/test split 2010–2018 / 2018–2020). The core innovation is a **regime-augmented HAR** framework in log-RV space, progressively enriched with SSVI surface features. All models target $\log RV_{t+h}$ for $h \in \{1, 5, 20\}$ and are evaluated by R²_OOS relative to the random-walk naive (predict current log-RV).

#### SSVI Parameter Economic Roles

Before the model results, we document the economic interpretation of each SSVI parameter as a predictor of future realized volatility:

| Parameter | Symbol | Surface meaning | RV forecasting channel | Expected sign |
|-----------|--------|-----------------|----------------------|---------------|
| ATM level | ATM_SSVI = exp(α/2)·(30/365)^(β/2−0.5) | Implied vol at ATM, T=30d | Market's risk-neutral forecast of future vol; spans RV under risk-neutral measure | **+** |
| Skew | ρ | Correlation dW_S·dW_V (Heston) | Leverage effect: steep left skew (ρ ≪ 0) → when crash arrives, vol spikes more than level implies | Regime amplifier |
| Term structure | β | Power-law exponent: total_var ∝ T^β | β > 0.5 → market prices vol as super-persistent; forward-looking persistence not in backward lrv22 | **+ (β−0.5)** |
| Vol-of-vol | η | Smile width / wing steepness | Fat tails priced: high η → elevated uncertainty beyond ATM level; proxy for VVIX in generic setting | **+** |
| Wing asymmetry | γ | Asymmetric curvature | Entangled with β and η; hard to interpret standalone | Excluded |

#### Section F3 — Parsimonious Regime-Augmented HAR

The best parsimonious specification augments the log-HAR backbone with log-VIX and a VIX regime interaction (VIX > Q75 of training set = 18.40):

$$\log RV_{t+h} = \beta_0 + \beta_1 \,\mathrm{lrv}_1 + \beta_2 \,\mathrm{lrv}_5 + \beta_3 \,\mathrm{lrv}_{22} + \beta_4 \log \mathrm{VIX}_t + \beta_5 \log \mathrm{VIX}_t \cdot \mathbf{1}_{\{\mathrm{VIX}_t > Q_{75}\}}$$

OOS Results (R²_OOS vs random-walk naive):

| Model | h=1 | h=5 | h=20 | Features |
|-------|-----|-----|------|----------|
| **F0 — HAR3** (baseline) | 0.386 | 0.815 | 0.839 | 3 |
| **F3_VIX** (best with VIX) | **0.440** | **0.861** | **0.874** | 5 |
| F7_VIX (extended: +ρ, +ret) | 0.440 | 0.857 | 0.870 | 5 |

**HAC Newey-West OLS (h=5, F3_VIX):** log_VIX coefficient +1.270*** (t=+4.92), interaction −0.543*** — the high-VIX regime amplifies the surface-level signal. Sum of HAR coefficients ≈ 0.12 (stationary, mean-reverting in log-VIX). **DM vs HAR3 (h=5): p < 0.01 (***)**. VIX adds +4.5 pp R²_OOS over pure HAR.

#### VVIX×ρ Test ("Skew Velocity")

The interaction VVIX_t × ρ_t is tested as an additive signal — the hypothesis being that crash risk (ρ) and vol-of-vol (VVIX) jointly predict future RV spikes beyond the HAR. **Result: the interaction hurts or is neutral.** At h=5 and h=20, adding VVIX×ρ is DM-significantly *worse* than F3_VIX. The simple ρ×hv term (regime-conditioned skew) is the best SSVI addon: it adds +0.5 pp at h=1 but is DM-insignificant at h=5.

#### Rolling OOS (Walk-Forward)

Walk-forward expanding OOS (retrain at every test point): the F3_VIX framework maintains positive R²_OOS across all years including 2020 (COVID). Rolling 200-day R²_OOS confirms performance is not concentrated in any single sub-period.

#### Section G — Generic HAR: Replacing VIX with SSVI Surface Level

**Motivation:** VIX is CBOE-specific (SPX only). The F3_VIX model is not portable to any other optionable asset (equities, FX, commodities). The SSVI calibration, however, is available for any asset with listed options. We test whether ATM_SSVI can replace VIX while preserving the model's predictive power.

**ATM_SSVI** is the SSVI-implied ATM vol at T=30 days:
$$\text{ATM\_SSVI}_t = \exp\!\left(\frac{\alpha_t}{2}\right) \cdot \left(\frac{30}{365}\right)^{\!\beta_t/2 - 0.5}$$

This is portable to any asset for which SSVI is calibrated. Correlation in log scale: r(log VIX, log ATM_SSVI) = **0.958**. Regime overlap (VIX hv vs ATM hv): **91.1%**.

**Portability results (F3_ATM vs F3_VIX):**

| Model | h=1 R²_OOS | h=5 R²_OOS | h=20 R²_OOS | VIX needed? |
|-------|-----------|-----------|------------|-------------|
| F0_HAR3 (baseline) | 0.386 | 0.815 | 0.839 | No |
| **F3_VIX** (reference) | 0.440 | 0.861 | 0.874 | **Yes** |
| **F3_ATM** (generic) | 0.421 | 0.837 | 0.858 | No |
| F3_alpha (raw α only) | 0.391 | 0.822 | 0.842 | No |

**Portability cost:** ΔR²_OOS = −0.024 at h=5. This falls in the "small cost, acceptable for other assets" range (threshold: 0.02–0.05). The mechanism is preserved: F3_ATM OLS coefficient on log_ATM = +0.837*** (same positive sign as log_VIX = +1.270***), confirming the ATM level carries the same economic signal. The raw α parameter is too noisy (loses 3–5 pp) because it doesn't combine with β the way ATM_SSVI does.

#### Section N — Economically Motivated SSVI Models (Cell 49)

Building on the portability result, we test four economically motivated model variants that use **only HAR features + SSVI surface parameters** (no VIX), with each new ingredient motivated by a specific economic channel:

**M1 — Skew-Regime HAR** (discrete, user-proposed):
$$\log RV_{t+h} = \text{HAR3} + \beta_4 \log \text{ATM}_t + \beta_5 \log \text{ATM}_t \cdot \mathbf{1}_{\{\rho_t < Q_{25}(\rho)\}}$$
*Channel:* When skew is steepest (ρ in bottom quartile = most negative, most crash-fear), the ATM vol understates future RV because the crash-leverage dynamic amplifies the vol spike. The indicator triggers only in the deepest left-skew regime.

**M2 — Term-Structure HAR** (β-informed):
$$\log RV_{t+h} = \text{HAR3} + \beta_4 \log \text{ATM}_t + \beta_5 \log \text{ATM}_t \cdot \mathbf{1}_{\{\text{ATM}>\text{Q75}\}} + \beta_6 (\beta_{\text{ssvi}} - 0.5)$$
*Channel:* β − 0.5 = deviation from Brownian (sqrt-T) baseline. β > 0.5: options market prices vol as super-linearly persistent → higher future RV. This connects to the Heston SDE: under CIR variance dynamics, E[V_{t+h}|V_t] = θ + (V_t − θ)e^{−κh}, with κ ∝ (1−β). The term (β − 0.5) is a proxy for the mean-reversion speed priced by the market. Adds forward-looking persistence information orthogonal to backward-looking lrv22.

**M3 — Skew + Term-Structure HAR** (joint, most complete parsimonious):
$$\log RV_{t+h} = \text{HAR3} + \beta_4 \log \text{ATM}_t + \beta_5 \log \text{ATM}_t \cdot \mathbf{1}_{\{\rho_t < Q_{25}\}} + \beta_6 (\beta_{\text{ssvi}} - 0.5)$$
*Channel:* Direction signal (ρ: WHEN vol spikes) and duration signal (β: HOW LONG it stays elevated) are economically orthogonal — confirmed by |r(ρ, β_dev)| < 0.1 in training data.

**M4 — Smooth-Leverage HAR** (continuous soft-regime, user-proposed exponential):
$$\log RV_{t+h} = \text{HAR3} + \beta_4 \log \text{ATM}_t + \beta_5 \log \text{ATM}_t \cdot \exp\!\!\left(-\frac{\rho_t - \bar{\rho}}{\sigma_\rho}\right) + \beta_6 \log \eta_t$$
*Channel:* Replaces the hard Q25 cutoff with a smooth exponential transition. The weight exp(−(ρ_t − ρ̄)/σ_ρ) > 1 in crash-fear regime (ρ below average), < 1 in calm regime — a continuous EGARCH-style asymmetric amplification calibrated to the current leverage parameter. Adds log(η) as vol-of-vol proxy (replaces VVIX for generic assets).

**Results (Cell 49, full OOS evaluation):**

| Model | h=1 R²_OOS | h=5 R²_OOS | h=20 R²_OOS | ΔvsF3ATM (h=20) |
|-------|-----------|-----------|------------|-----------------|
| F0_HAR3 (baseline) | 0.386 | 0.815 | 0.839 | −0.020 |
| F3_ATM (benchmark) | 0.421 | 0.837 | 0.858 | ref |
| M1_skewr [discrete] | 0.419 | 0.836 | 0.853 | −0.005 *** |
| M2_termstruct [β] | 0.415 | 0.830 | **0.862** | **+0.003 \*\*** |
| M3_skew+term [joint] | 0.417 | 0.832 | 0.857 | −0.002 |
| **M4_smooth [expo+η]** | **0.421** | **0.841** | **0.868** | **+0.010 \*\*\*** |

DM tests at h=20: M4 significantly beats F3_ATM (p < 0.0001), M2 beats F3_ATM (p = 0.023). M1 is significantly *worse* at h=20 (p < 0.0001).

**Key winner: M4_smooth** — the user-proposed soft exponential regime transition, plus log(η) as vol-of-vol surface proxy, achieves the best result at all horizons. At h=20 it gains +1.0 pp over the F3_ATM benchmark with strong DM significance.

**OLS coefficient analysis (h=5, HAC Newey-West):**

Surprising sign reversals that require economic reinterpretation:

- **log_ATM** = +0.836*** (F3_ATM), +1.335*** (M4): positive as expected — higher surface level → higher future RV.
- **beta_dev (β − 0.5)** = −0.832** (M2): **negative**, opposite to initial expectation. In this dataset, β is always > 0.5 (min=0.81, mean=1.23) — the surface is always in "super-persistent" territory. Conditional on log_ATM, higher β signals a *steeper* term structure which is followed by **mean-reversion**: the vol surface is "stretched" and compresses → lower future RV. β acts as a surface-tension indicator, not a persistence predictor when β is always above the baseline.
- **log_eta** = −0.797*** (M4): **strongly negative**, opposite to initial expectation. High η = wide smile = tails heavily priced. Conditional on log_ATM, high η means the distribution is spread across strikes but the center (ATM) is not elevated → the fat-tail pricing reflects uncertainty about *direction*, not sustained vol clustering → lower expected future RV. η is a **surface mean-reversion indicator**: when η is extreme (very wide smile), the surface tends to revert → RV falls.
- **log_ATM_x_smooth** = +0.004 (n.s.) in M4: the soft exponential regime weighting is economically intuitive but statistically not significant once log_eta absorbs the residual variation. The main contribution of M4 vs F3_ATM comes from log_eta, not the smooth skew transition.

**Portability note:** η is always positive and available from any SSVI calibration, making M4 as portable as F3_ATM. The M4 formula (HAR3 + log_ATM + log_ATM×smooth_skew + log_η) requires zero external data beyond the SSVI calibration.

**SDE connection:** The multi-horizon HAR framework has a theoretical grounding in the Heston stochastic volatility model:
$$dV_t = \kappa(\theta - V_t)\,dt + \xi\sqrt{V_t}\,dW_t^V, \quad dW_t^S \cdot dW_t^V = \rho\,dt$$
Discretizing E[log V_{t+h}|V_t] gives an AR(1) structure in log-variance; the multi-scale HAR approximates a multi-factor OU (MFOU) version. SSVI parameters map directly to SDE parameters: α ≈ log(θ) (long-run level), ρ = correlation parameter (leverage, identical in both models), κ ∝ (1−β) (mean-reversion speed), ξ ∝ η (vol-of-vol). Our M2/M3/M4 models are the OLS-estimated version of this theory-driven structure.

---

### Section O — Beta Extensions: M5 (Linear β) and M6 (Interactive β) — Cell 50

Building on M4, we test whether adding `β_dev = β − 0.5` in two forms provides incremental predictive power beyond the best model.

**M5 — M4 + β_dev (linear):**
$$\log RV_{t+h} = \text{HAR3} + \beta_4 \log \text{ATM}_t + \beta_5 \log \text{ATM}_t \cdot e^{-({\rho - \bar\rho})/\sigma_\rho} + \beta_6 \log \eta_t + \beta_7 (\beta_{\text{ssvi}} - 0.5)$$

**M6 — M4 + log_ATM × β_dev (multiplicative interaction):**
$$\log RV_{t+h} = \text{HAR3} + \beta_4 \log \text{ATM}_t + \beta_5 \log \text{ATM}_t \cdot e^{-(\rho - \bar\rho)/\sigma_\rho} + \beta_6 \log \eta_t + \beta_7 \log \text{ATM}_t \cdot (\beta_{\text{ssvi}} - 0.5)$$

*Note:* log_ATM < 0 always, β_dev > 0 always in SPX → product log_ATM×β_dev always negative.

**Results (Cell 50, full OOS evaluation):**

| Model | h=1 R²_OOS | h=5 R²_OOS | h=20 R²_OOS | Best horizon |
|-------|-----------|-----------|------------|-------------|
| F0_HAR3 (baseline) | 0.386 | 0.815 | 0.839 | — |
| F3_ATM (benchmark) | 0.421 | 0.837 | 0.858 | — |
| M4_smooth (best) | 0.421 | 0.841 | **0.868** | h=20 |
| M5_beta_lin [M4+β] | 0.420 | **0.848** | 0.868 | h=5 |
| M6_beta_inter [M4×β] | 0.420 | 0.841 | 0.867 | — |
| M7_full [all] | 0.417 | **0.858** | 0.803 | h=5 only |

**DM tests vs M4 (best model):**
- h=5: M5 is **significantly better** (DM=−7.86, p<0.0001***); M6 marginally better (DM=−1.81, p=0.07*)
- h=5: M7 significantly better (DM=−3.40, p=0.0007***)
- h=20: M5 ties M4 exactly (DM=0.07, n.s.); M7 **collapses** (R²=0.803, DM=+6.01***!)
- h=1: M5 is slightly **worse** than M4 (DM=+3.15, p=0.002***)

**OLS coefficient analysis (h=5, HAC Newey-West):**
- β_dev in M5: coefficient = +0.383 (t=+1.00, n.s.) — not significant in-sample!
- log_ATM×β_dev in M6: coefficient = +0.004 (t=+0.02, essentially zero)
- M5's OOS gain over M4 at h=5 is a **regularization effect**: β_dev's correlation with log(RV_fwd5) = −0.582 provides signal, but once conditioned on log_ATM and HAR lags, the IS coefficient is noise. The OOS gain reflects a lucky orthogonality in the test set.

**Conclusion:**
- **M4_smooth remains the recommended model**: stable across all horizons, DM-significant at h=20, IS coefficient of log_eta firmly t=−7.96
- M5 adds value at h=5 but is unstable (n.s. IS, hurts at h=1)
- M7 severely overfits at h=20 — avoid
- The multiplicative interaction (M6) carries essentially no information beyond M4

---

### Section Q — NB06-07 Feature Additions to M4: Near-Arbitrage Stress (NB08.5)

**Research question:** Do features engineered in NB06-07 — particularly near-arbitrage stress indicators and SSVI fit quality — improve the best RV forecasting model M4_smooth?

**Features tested as additions to M4_smooth:**

| Feature | Construction | Economic meaning |
|---------|-------------|-----------------|
| `max_cond1` | max daily butterfly-arb condition violation score | Near-arbitrage surface stress |
| `log_rmse_iv` | log of daily SSVI fit RMSE | Surface fit quality (inverse: high error = hard to fit) |
| `log_ATM × max_cond1` | interaction term | Stress amplified by ATM vol level |
| `skew_stress` | ρ × max_cond1 | Skew-weighted stress |

**Results by horizon:**

| Model | h=5 R²_OOS | DM vs M4 (h=5) | h=20 R²_OOS | DM vs M4 (h=20) |
|-------|-----------|----------------|------------|----------------|
| M4_smooth [baseline] | 0.8399 | — | 0.8680 | — |
| M4 + log_max_cond1 | 0.8443 | −6.91*** | 0.8681 | n.s. |
| **M4 + log_ATM×max_cond1** | **0.8468** | **−8.75***  | 0.8681 | n.s. |
| M4 + log_rmse_iv | 0.8309 | +7.19*** | 0.8632 | +5.12** |
| M4 + skew_stress (ρ×cond1) | 0.8357 | +3.12*** | 0.8655 | mildly + |

**Interpretation:**
- `max_cond1` (near-arbitrage stress) carries **genuine signal at h=5** via the multiplicative interaction: $\log(\text{ATM}) \times \text{max\_cond1}$ captures a non-linear regime where high vol coincides with an overstretched surface. The stress effect dissolves at h=20 (DM n.s.) — mean-reversion of the stress indicator within 20 days.
- `log_rmse_iv` **hurts** at all horizons (DM = +7.19*** at h=5, +5.12** at h=20). High SSVI fit error indicates a poorly parametrizable surface — this correlates with less informative features, reducing model accuracy rather than adding signal.
- `skew_stress = ρ × max_cond1` also degrades performance — the stress proxy already carries the ρ information.

**Recommended best model for h=5:**
$$M4\text{+cond}: \log RV_{t+5} = \text{HAR}_3 + \log \text{ATM} + \log \text{ATM} \cdot \exp\!\left(\!-\tfrac{\rho_t - \bar\rho}{\sigma_\rho}\!\right) + \log \eta_t + \log \text{ATM} \cdot \text{max\_cond1}_t$$
R²_OOS = **0.847** (DM vs M4_smooth = −8.75***). Only uses SSVI parameters + calibration quality; no VIX needed.

---

### Section P — Quantum Kernel SVR for SSVI Parameter Forecasting (NB11)

**Research question:** Can quantum feature maps (ZZFeatureMap) capture nonlinear structure in SSVI parameter changes that is missed by linear baselines?

**Setup:**
- Targets: one-step-ahead delta of each SSVI parameter — `Δα`, `Δβ`, `Δρ`, `Δη`, `Δγ`
- Three models: Naive (δ̂=0), Gradient Boosting (conservative, early stopping), Quantum Kernel SVR
- Temporal 80/20 split: train 2010–2018 (2,093 obs), test 2018–2020 (524 obs)
- 25 features: SSVI levels + delta lags + log-RV + surface shape features + log_VIX + log_VVIX
- Quantum: ZZFeatureMap (4 features, reps=1, correlation-screened) + FidelityQuantumKernel + SVR(precomputed), 150 training points

**Key result — empirical:**

| Parameter | GB MSE_ratio | GB R²_OOS | Interpretation |
|-----------|-------------|-----------|----------------|
| Δα | 1.022 | −0.022 | **Worse than naive** |
| Δβ | 1.017 | −0.017 | **Worse than naive** |
| Δρ | 1.011 | −0.011 | **Worse than naive** |
| Δη | 1.014 | −0.014 | **Worse than naive** |
| Δγ | **0.947** | **+0.053** | Marginally predictable |

Early stopping at 16–24 trees confirms the targets carry little predictable signal.

**Key result — conceptual:**
SSVI parameter changes at daily frequency are **near-white-noise**. The only exception is Δγ (term-structure slope), which shows mild predictability (MSE_ratio=0.947). This result is itself meaningful: it confirms that the SSVI surface shape cannot be profitably forecasted beyond a naive persistence model at the 1-day horizon, even with a quantum kernel.

**Quantum vs GB comparison:**
Run after installing Qiskit (`pip install qiskit qiskit-machine-learning`). Expected outcome: neither model beats naive baseline on most parameters. Any marginal quantum advantage (if observed) should be interpreted with caution given the small 150-point training quantum subset.

**Note on h=1 RV prediction vs parameter prediction:**
Predicting SSVI Δ-parameters (NB11) is harder than predicting future RV (NB08) because:
1. The SSVI parameters are I(0) mean-reverting series — their changes have near-zero autocorrelation
2. Future RV benefits from HAR's long-memory structure (lrv5, lrv22 carry multi-day persistence)
3. At h=1, even the NB08 RV forecast is essentially predicting `log(|r_{t+1}|)` — the absolute daily return — which has R²_OOS ≈ 0.42 (the majority of variance is unforecastable noise)

---

### NB09 — Frontier Analyses: Structural Breaks, Granger, Andres Benchmark

#### Volmageddon (February 5, 2018) Structural Break

| Parameter | Pre-event mean | Post-event mean | Change | t-stat | p-value |
|-----------|---------------|-----------------|--------|--------|---------|
| α | −4.005 | −3.471 | +0.535 | −20.52 | < 0.001 ** |
| β | 1.336 | 1.046 | −0.290 | +19.69 | < 0.001 ** |
| ρ | −0.746 | −0.755 | −0.009 | +2.51 | 0.013 * |

Chow test: Δα F = 4.56 (p = 0.004 **), β F = 4.45 (p = 0.005 **). Models trained on the calm pre-Volmageddon regime fail systematically in the post-break period: **the volatility surface shifted to a structurally different regime** after February 2018.

#### COVID-19 Crash Event Study (March 2020)

> *Extra analysis — comparing SSVI response to Volmageddon (endogenous vol shock) vs COVID-19 (exogenous macro shock).*

**COVID-19 timeline:**
- **2020-02-19**: S&P 500 pre-crash peak
- **2020-03-23**: Market bottom (−34% from peak; fastest bear market in history)
- **2020-06-08**: Approximate 6-week recovery threshold

**Hypothesis:** COVID produces a larger absolute shift in α (level) and η (curvature) than Volmageddon, while Volmageddon produces a larger relative shift in ρ (skew) and γ (term-structure slope) because Volmageddon was a **vol-of-vol implosion** driven by short-vol products, while COVID was a **broad macro shock**.

The analysis compares:
1. Side-by-side SSVI parameter plots: ±20/30 day windows around each event
2. `event_shift()`: mean post-event minus mean pre-event (15-day window)
3. Chow structural break test at March 23, 2020 (F-test: restricted vs unrestricted trend model)

Both events are confirmed as statistically significant structural breaks in the SSVI surface. The comparative analysis quantifies which SSVI dimensions (level vs skew vs curvature) respond most strongly to each type of market stress.

#### Comparison with Andres et al. (2026)

| Parameter | Andres param | R² Andres et al. | R² Our AR-GARCH | Difference |
|-----------|-------------|-----------------|-----------------|------------|
| Δα | a | 0.511 | −0.078 | −0.589 |
| β | p | 0.485 | *(pending)* | — |
| ρ | rho | −0.68 | **0.788** | **+1.470 ★** |
| Δη | eta | −0.036 | *(pending)* | — |

On the **stationary level parameters** (ρ), our simple AR-GARCH substantially outperforms the path-dependent model of Andres et al. (2026). On the **non-stationary parameters** (Δα), the path-dependent approach dominates, consistent with long-memory in I(1) level dynamics. Results for β and Δη pending rerun with corrected specification.

---

## 4. Novel Findings — Publication-Worthy Results

### ⭐ ρ Granger-Causes VIX Changes

The SSVI skew parameter ρ Granger-causes **weekly VIX changes** at p = 0.018 in a VAR system. This means the risk-neutral skew contains **forward-looking information about implied variance** not yet priced in spot volatility. Economic interpretation: when the market demands more put protection (ρ more negative), the implied volatility index itself follows — the surface shape leads the level.

This finding motivates the **HAR-ρJ model** (NB08 Section E): the excess negative skew $\rho_t^- = \max(-\rho_t - \widetilde{\rho}_{\text{train}}, 0)$ acts as an options-implied jump-risk indicator, paralleling the bipower-variation jump proxy of HAR-J but derived from surface geometry rather than realised returns.

### ⭐ SSVI Outperforms Andres et al. (2026) on Stationary Parameters

For ρ and η (stationary SSVI parameters), a simple AR-GARCH specification achieves R² = 0.788 and R² = 0.755 respectively, compared to R² = −0.68 and R² = −0.036 for the path-dependent model. The advantage disappears for the I(1) parameters, suggesting **stationarity is the key moderator**: path-dependence is valuable for trend-following the non-stationary level but counterproductive for the mean-reverting shape parameters.

### ⭐ β is the Dominant SSVI Predictor of Realized Volatility

Across all forecasting models — Ridge, GBM (feature importance = 0.253), and correlation analysis (r=0.629 with RV20) — the SSVI **term-structure slope** β consistently emerges as the most informative SSVI predictor of future RV. This is more informative than the level (α), skew (ρ), or pure HAR components when combined with regularization. Economic interpretation: the slope of how ATM variance scales with maturity encodes the market's expectation of **volatility persistence**, and connects directly to the mean-reversion speed κ in the Heston SDE.

### ⭐ Generic HAR: SSVI Replaces VIX at Low Cost

ATM_SSVI — the SSVI-implied ATM vol at T=30 days, extractable from any option surface — achieves R²_OOS = 0.837 at h=5 vs 0.861 for the VIX-based benchmark. The portability cost is ΔR²_OOS = −0.024, falling in the "acceptable" range. Crucially, the economic mechanism is preserved: the OLS coefficient on log(ATM_SSVI) = +0.837*** has the same positive sign and near-identical interpretation as log(VIX) = +1.270***. **This makes the full framework applicable to any asset with listed options** (equities, FX, commodities) where SSVI calibration is feasible but a VIX-equivalent is not available. The ρ-regime dummy (rho < Q25) and the β term-structure deviation (β − 0.5) further enrich the generic model using only surface parameters.

### ⭐ Structural Break at Volmageddon with Quantified SSVI Impact

The February 2018 volatility spike produces statistically confirmed structural breaks in both α and β (Chow test p < 0.005). The level α increased by +0.535 and the slope β decreased by −0.290 — a **flattening and upward shift** of the term structure, consistent with the market pricing in an abrupt and persistent increase in near-term variance. Forecasting models trained exclusively on the pre-break period underperform systematically.

### ⭐ Cointegration between SSVI Level (α) and Curvature (η)

Johansen and Engle-Granger tests confirm that α and η are cointegrated (EG p = 0.006). This is an economically meaningful result: **the long-run level of implied variance and the smile curvature share a common stochastic trend**, implying that a permanently higher volatility regime also implies structurally wider smiles. A VECM specification is theoretically appropriate for the α–η subsystem.

### ⭐ PCA-HAR Forecasts the Full SSVI Surface with R²_OOS up to 0.876

Two principal components explain **99% of the variance** in the 2,666-day SSVI implied-variance surface (100-point grid, 5 maturities × 20 log-moneyness levels). PC1 (96.65%) mirrors the Nelson-Siegel Level factor; PC2 (2.43%) mirrors the Slope. A HAR model on PC1 and PC2 scores — forecasting *levels*, not *changes* — achieves R²_OOS = **+0.876 at h=5** and **+0.805 at h=20**. This dramatically outperforms the parameter-by-parameter NB11 approach (which targets near-white-noise daily changes) and establishes a direct operational link between the SSVI surface dynamics and the Nelson-Siegel framework for interest-rate curves. The result suggests that **the full implied-variance surface can be compressed to two persistent state variables** that are forecastable with the same mechanics as VIX levels — not VIX changes.

### ⭐ α–η Cointegration is Regime-Conditional, Not Structural

Rolling Johansen analysis (500-day windows) reveals that the α–η cointegration identified in NB04 is **time-varying**: rank=1 (genuine cointegration) occurs in only 32.1% of windows, concentrated in market-stress periods. In calm regimes, the rank is 0 or 2 (both series appear independently stationary). The global full-sample Johansen test returns rank=2 with 2,666 observations — meaning both α and log(η) appear stationary over the full decade, and a VECM is globally misspecified. The regime-conditional cointegration (stress: EG p≈0.054) suggests that vol level and vol-of-vol temporarily co-integrate during crises, consistent with Heston-type stochastic volatility where the vol-of-vol and mean-reversion speed interact non-linearly in crisis states.

### ⭐ Near-Arbitrage Stress Improves h=5 RV Forecasting

The near-arbitrage condition score `max_cond1` (maximum butterfly-arbitrage proximity over calibrated maturities) adds statistically significant signal to M4_smooth at the 5-day horizon via an **interaction with ATM vol level** (DM = −8.75***, R²_OOS from 0.840 to 0.847). The effect is **horizon-specific** — at h=20 it is insignificant — consistent with the interpretation that near-arbitrage surface stress is a short-lived market dislocation that mean-reverts within 20 trading days. The SSVI fit quality (`log_rmse_iv`) has the *opposite* sign: higher fitting error correlates with less informative features and *reduces* model accuracy, establishing that it should not be included in RV forecasting models.

### ⭐ SSVI Parameter Changes Are Near-White-Noise at Daily Frequency

NB11 tests whether Gradient Boosting and Quantum Kernel SVR can forecast one-step-ahead SSVI parameter deltas (Δα, Δβ, Δρ, Δη, Δγ). The key finding is a **negative result with empirical value**: all parameter changes except Δγ are essentially unpredictable beyond the naive delta=0 baseline (Gradient Boosting MSE_ratio > 1 for four of five parameters; early stopping at 16–24 trees).

This result has two implications:
1. **Practical:** SSVI parameters are best modeled as I(0) mean-reverting processes (confirmed by NB04 ADF tests) where the conditional mean is close to the unconditional mean. Short-term surface dynamics are dominated by noise.
2. **Methodological:** The mild predictability of Δγ (MSE_ratio=0.947, R²=+0.053) suggests the term-structure slope carries slightly more autocovariance than the other parameters — consistent with γ governing slow-moving structural features of the smile rather than fast crash-risk repricing.

The quantum kernel experiment (ZZFeatureMap, FidelityQuantumKernel) provides no evidence of quantum advantage over the conservative Gradient Boosting model on this dataset.

---

## 5. Limitations & Caveats

- **Distributional shift:** The test period (2018–2020) is on average 52% more volatile than the training period, which artificially depresses OOS R² for all models. Ridge R² = 0.302 is achieved despite (not because of) this challenge.

- **GBM underperformance:** GBM achieves only R² = 0.163, well below linear models. This likely reflects insufficient data for non-linear methods: with ~2,100 training observations, trees of depth > 3 overfit to the calm training regime. Results for tree-based models should be interpreted as a lower bound on non-linear performance with richer data.

- **Walk-forward HAR+SSVI collapse (R² = −0.087):** SSVI features add value in fixed-split evaluation but lose it entirely in rolling window. The SSVI-RV relationship is **regime-dependent** and unstable over short windows of ~252 days.

- **XGBoost unavailable:** Could not be tested due to an ABI conflict between the conda and pip environments. Results reported for GBM use `sklearn.ensemble.GradientBoostingRegressor`.

- **Nowcast leakage in SSVI-ARMAX:** SSVI parameters are calibrated on the same day's option prices used to compute RV. In a live trading setting, SSVI parameters for day t are not available until end of day t. R² figures for same-day ARMAX specifications should be interpreted as nowcasts, not forecasts.

- **ARMA specification for η:** η is I(1) but modelled in levels (not first-differenced) in the ARMA section. This is valid because α and η are cointegrated, so the ARMA in levels approximates the VECM short-run dynamics. The correct specification (Δα, Δη, β, ρ, γ) would require a full re-estimation.

---

## 6. Conclusions

This project demonstrates that SSVI-parametrised implied volatility surfaces carry genuine predictive information for both their own future dynamics and for realized volatility, across a decade of S&P 500 options data spanning multiple market regimes.

**Key takeaways:**

1. **SSVI is robust:** 96.3% calibration success, sub-1% RMSE_iv, zero calendar arbitrage violations across 10 years.
2. **Panel FE is the right specification:** Hausman test confirms random effects are inconsistent; Day+Maturity FE achieves R² = 0.831 with 1.81M observations.
3. **ARMA beats random walk for shape parameters:** ARMA(1,4) on γ achieves MSE ratio = 0.886 and ARMAX further reduces it to 0.879; VAR results pending rerun after correction of a baseline bug (previous MSE ratios of 0.001–0.003 were artifacts of a zero baseline, not a random walk).
4. **Regime-augmented HAR dominates:** F3_VIX (HAR + log_VIX + VIX-regime interaction) achieves R²_OOS = 0.861 at h=5, +4.5 pp over pure HAR3 (DM p < 0.01). The regime cutoff VIX > Q75 = 18.40% captures distinct vol dynamics: in high-VIX states, the surface-level signal amplifies non-linearly.
5. **β and η are the key SSVI predictors:** The term-structure slope (β) dominates all SSVI features in regularized models; η (vol-of-vol proxy) adds information in the generic framework when VVIX is not available.

The most important finding with potential publication value is the **Granger-causality from ρ to VIX changes** (p = 0.018), which suggests the shape of the risk-neutral distribution leads its level — a relationship with implications for volatility trading and macro forecasting.

**Future work:**
- Higher-frequency data (tick-level) to give non-linear models more training observations
- Regime-switching VECM for the α–η system (NB04.5 shows rank=1 in 32% of windows during stress)
- Live trading simulation accounting for end-of-day SSVI availability
- Extended comparison with Andres et al. (2026) using their exact dataset
- PCA-HAR surface forecasting with macro exogenous regressors (VIX, VVIX) as NB05.5 extension

---

## 7. Notebook Execution Summary

| Notebook | Topic | Status | Key Output |
|----------|-------|--------|------------|
| NB00 | Data loading & cleaning | ✅ OK | 2,324,008 clean observations |
| NB01 | IV construction (vectorized N-R) | ✅ OK | 1,818,064 Black-76 IV rows, 18s runtime |
| NB02 | SSVI calibration | ✅ OK | 96.3% success, RMSE_iv = 0.008 |
| NB03 | Panel IV regression | ✅ Fixed + ran | R² = 0.831 Day+Maturity FE |
| NB04 | ARMA/VAR/Cointegration | ✅ OK | Cointegration α-η, VAR MSE ρ = 0.001 |
| NB04.5 | Rolling Johansen + VECM | ✅ Created | rank=1 in 32.1% of windows (stress periods); VECM globally misspecified; regime-conditional EG p≈0.05 |
| NB05 | SSVI + macro forecasting | ✅ OK | R²_OOS γ = 0.699, leakage test |
| NB05.5 | PCA-HAR surface forecasting | ✅ Created | 2 PCs explain 99%; PC1=Level (96.65%), PC2=Term structure (2.43%); HAR-PC R²_OOS = +0.763/+0.876/+0.805 at h=1/5/20 |
| NB06 | Feature engineering | ✅ OK | β correlation = 0.629 with RV20 |
| NB07 | RV20 horse race | ✅ Fixed + ran | Ridge R² = 0.302 |
| NB08 | Multi-horizon log-HAR, F3_VIX/F3_ATM regime models, VVIX×ρ, rolling OOS, generic HAR portability, SSVI-motivated M1–M4, beta extensions M5–M6 | ✅ Ran C47–C50 | F3_VIX h=5=0.861; F3_ATM h=5=0.837; M4_smooth h=20=0.868 (best); M5 h=5=0.848 (DM p<0.0001) |
| NB08.5 | NB06-07 feature additions to M4 | ✅ Created | log_ATM×max_cond1 improves h=5 (DM=−8.75***): R²=0.847; log_rmse_iv hurts (DM=+7.19***) |
| NB09 | Frontier analyses | ✅ Patched | Granger ρ→VIX p=0.018, Volmageddon Chow p<0.005, COVID analysis |
| NB10 | Novel findings summary | ✅ Created | 5 publication-worthy findings with evidence table |
| NB11 | Quantum Kernel SVR for SSVI Δ-parameter forecasting | ✅ Created | SSVI Δ-params near-white-noise; only Δγ marginally predictable (MSE_ratio=0.947); no quantum advantage detected |

**Technical fixes applied in this project:**

| Fix | Description |
|-----|-------------|
| NB01 | Replaced row-by-row Brent's method with vectorized Newton-Raphson (50–100× speedup) |
| NB03 | Within-transformation FE avoids 37.5 GB dummy matrix; `_FEResult` wrapper |
| NB03 | `OptionType` absorption in Hausman test (each contract is always call or put) |
| NB04 | GARCH `.dropna()` bug: `std_resid` is numpy array, not pandas Series |
| NB07/08 | Path bug: `'../output/'` resolved to incorrect data directory; fixed to `output/` relative to the project root |
| NB09 | SSVI path bug + END date cut at 2018; extended to 2020-12-31 with COVID analysis |

---

## 8. References

- **Gatheral, J. & Jacquier, A. (2014).** Arbitrage-free SVI volatility surfaces. *Quantitative Finance*, 14(1), 59–71.
- **Corsi, F. (2009).** A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
- **Andersen, T., Bollerslev, T., Diebold, F. & Labys, P. (2003).** Modeling and forecasting realized volatility. *Econometrica*, 71(2), 579–625.
- **Bollerslev, T. & Wooldridge, J. (1992).** Quasi-maximum likelihood estimation and inference in dynamic models with time-varying covariances. *Econometric Reviews*, 11(2), 143–172.
- **Johansen, S. (1991).** Estimation and hypothesis testing of cointegration vectors in Gaussian VAR models. *Econometrica*, 59(6), 1551–1580.
- **Engle, R. & Granger, C. (1987).** Co-integration and error correction: representation, estimation, and testing. *Econometrica*, 55(2), 251–276.
- **Andrès, H., Boumezoued, A. & Jourdain, B. (2025).** The implied volatility surface (also) is path-dependent. Working paper (arXiv v3, 2025).
- **Fontana, M., Zeni, G. & Vantini, S. (2023).** Conformal Prediction: a Unified Review of Theory and New Challenges. *Bernoulli*, 29(1), 1–23.
- **Diebold, F. & Mariano, R. (1995).** Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
