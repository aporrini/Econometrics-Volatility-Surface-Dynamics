# SSVI Implied-Volatility Surface Dynamics & Realized-Volatility Forecasting
## Evidence from S&P 500 Options (2010–2020)

**Econometrics Project — Academic Year 2025/26**
Politecnico di Milano — Insurance & Econometrics
Authors: Alessio Porrini, Marco Amarilli, Camilla Introzzi, Christian Frigerio

---

**Abstract.** Using ~1.5 million filtered S&P 500 option quotes (2010–2020), we calibrate a daily arbitrage-free SSVI implied-volatility surface and pursue two aims. First, we ask whether the surface's structural information improves the time-series modelling and forecasting of (i) its own parameters and (ii) realized volatility (RV), relative to standard benchmarks (random walk, HAR) — with the deliberately asset-agnostic twist of testing whether an underlying's *own* option surface can replace the SPX-specific VIX as a regime signal portable to any optionable asset. Second, we extend the Avellaneda & Stoikov (2008) market-making framework to the full surface with an SSVI-based residual-risk add-on calibrated to a target Expected-Shortfall coverage. We deploy the complete econometric toolbox (panel regression with fixed/random effects and hypothesis testing, ARMA/ARMAX, VAR, non-stationarity and cointegration, point and probabilistic forecasting). The central result is *negative with informational content*: option-surface features do not beat parsimonious HAR for RV forecasting through the COVID-19 window, and the surface behaves as a near-efficiently-priced object at the daily frequency. The two original contributions — a portable RV-forecasting study (Notebook C) and a surface market-making risk engine (Notebook D) — are the project's main value beyond the required techniques.

---

## 1. Introduction and Literature Review

**Research aims.** The project pursues two connected aims. *(i) Predictive content / market efficiency:* does the structural information in an arbitrage-free IV surface carry incremental out-of-sample (OOS) predictive content for (a) its own future dynamics and (b) future realized volatility, beyond simple benchmarks (random walk, HAR)? A sharpened, deliberately asset-agnostic version of (b) asks whether a single underlying's *own* option surface can supply the regime information usually drawn from the VIX — which is specific to the S&P 500 — in a form that ports to any asset with a liquid option market. *(ii) Risk-management application:* can the **Avellaneda & Stoikov (2008)** optimal market-making framework, which sets quotes for a single instrument, be extended to the full implied-volatility surface with an SSVI-based residual-risk add-on calibrated to a target tail (Expected-Shortfall) coverage? The first aim is the efficient-pricing question; the second turns the surface model into a deployable quoting-and-risk tool.

**Economic framework.** Under efficient option pricing the no-arbitrage IV surface should be close to a martingale at short horizons: if daily surface increments were systematically predictable, a delta-hedged position would extract near risk-free profit, which a functioning market arbitrages away. The competing hypothesis — from the long-memory and path-dependence literature — is that volatility (realized *and* implied) carries persistent, multi-horizon structure that low-order autoregressive models miss but a long-memory model can exploit.

**Literature.** The methodological backbone rests on three works. We parametrise the surface with the SSVI model of **Gatheral & Jacquier (2014)**, which guarantees freedom from static (butterfly and calendar) arbitrage. The forecasting backbone is the Heterogeneous Autoregressive (HAR) model of **Corsi (2009)**, which approximates the long memory of realized volatility — documented by **Andersen, Bollerslev, Diebold & Labys (2003)** — with cascaded daily/weekly/monthly components. The hypothesis that the IV surface is itself *path-dependent*, and hence amenable to HAR-type modelling, is the recent contribution of **Andrès, Boumezoued & Jourdain (2025)**, which directly motivates applying HAR to the SSVI parameters. For the risk engine we build on **Avellaneda & Stoikov (2008)**. Supporting tools: **Johansen (1991)** and **Engle & Granger (1987)** for cointegration, **Bollerslev (1986)** GARCH for conditional heteroskedasticity, and the **Diebold & Mariano (1995)** test (small-sample corrected) for formal forecast comparison.

---

## 2. Data and Methodology

### 2.1 Dataset (primary focus)

**Source and composition.** The raw dataset is a panel of daily **S&P 500 European option** quotes spanning 2010-01-01 → 2020-12-31 (~2,769 trading files; ~3.33 million raw rows), each carrying strike, option type, bid/ask/mid, open interest and time-to-expiry (TTE). Zero-coupon **Treasury rates** (1M, 3M, 6M, 1Y, 2Y, 3Y) and the **VIX** are pulled from FRED; the SPX spot is from Yahoo Finance. The data are processed in three notebooks (`00_data_preparation`, `01_iv_dataset_construction`, `02_ssvi_calibration`).

**Cleaning pipeline (core technique #1).** Quality filters remove illiquid and economically meaningless quotes:

| Filter | Threshold |
|--------|-----------|
| Mid price | > \$0.05 |
| TTE | > 7 days |
| Bid-ask spread / mid | ≤ 30% |
| Log-moneyness | ∈ [−0.40, 0.30] |

The implied **forward** is extracted per maturity via put-call parity; Black-76 **implied volatility** is then computed by vectorised Newton-Raphson. The funnel is:

| Stage | Count |
|-------|-------|
| Raw rows | ~3,331,302 |
| After domain/liquidity filters | ~2,324,008 |
| Black-76 IV retained | ~1,500,012 |
| SSVI calibration success | 2,633 / 2,768 days (**95.1%**) |

**Derived parameter panel.** For each trading day we calibrate the SSVI surface, yielding a daily time series of the five parameters (α, β, ρ, η, γ). Calibration succeeds on **95.1%** of days with median RMSE_iv < 0.01 and **zero** calendar-arbitrage violations — this calibrated panel is the input to all downstream time-series analysis (Notebooks B and D). The realized-volatility target (Notebook C) is a daily log-RV proxy built from SPX returns.

### 2.2 SSVI parametrisation and parameter economics

$$\omega(k,\theta) = \frac{\theta}{2}\left\{1 + \rho\,\phi(\theta)\,k + \sqrt{(\phi(\theta)\,k+\rho)^2 + (1-\rho^2)}\right\}, \quad \phi(\theta) = \frac{\eta\,\theta^{-\gamma}}{1+\eta\,\theta^{1-\gamma}}$$

| Parameter | Economic role | Sample mean |
|-----------|--------------|-------------|
| α | log(ATM variance) — surface level | −3.45 |
| β | Term-structure slope | 1.19 |
| ρ | Skew / leverage effect | −0.77 |
| η | Vol-of-vol / smile width | 0.75 |
| γ | Term-structure decay rate | 0.54 |

### 2.3 Econometric toolbox — coverage of the core techniques

The project applies every core econometric technique of the course to the volatility data; the table maps each to its implementation. Section 3 mirrors this mapping, giving each method a self-contained results subsection (§3.1–§3.5), followed by the two original contributions (§3.6).

| # | Core technique | Implementation | Reported in |
|---|--------------------|----------------|-------------|
| 1 | Data cleaning, trend & seasonality | Cleaning funnel (NB 00–02); trend via I(d) classification + differencing; seasonality assessed and justified immaterial | §2.1, §3.1 |
| 2 | Linear regression + hypothesis testing | Panel IV regression with t/F-tests, RESET (NB A); OLS feature-significance pruning (NB C) | §3.2 |
| 3 | Fixed & random effects | Pooled OLS → Day FE → Day×Maturity FE → Surface-Cell FE, **Hausman test** FE-vs-RE (NB A) | §3.2 |
| 4 | ARMA models | BIC-optimal ARMA + **ARMAX** (exogenous VIX) on 10 series (NB B) | §3.4 |
| 5 | Point & probabilistic forecasting | Multi-horizon RV point forecasts + DM (NB C); **AR-GARCH 80% prediction intervals** (NB B); **ES₉₅ coverage** (NB D) | §3.5 |
| 6 | VAR models | VAR(2, BIC) on the 5-parameter system + **VECM** (NB B) | §3.4 |
| 7 | Non-stationarity & cointegration | ADF + KPSS, **Johansen** trace, **Engle-Granger**, rolling Johansen (NB B) | §3.3 |

**Evaluation metrics.** OOS skill is measured by $R^2_\text{OOS} = 1 - \sum_t (y_t-\hat y_t)^2 / \sum_t (y_t-y_{t-1})^2$ (relative to the naïve random-walk benchmark, so $R^2_\text{OOS}=0$ for the naïve model by construction), and pairwise comparison by a small-sample-corrected **Diebold-Mariano** test (Diebold & Mariano 1995; HAC Newey-West variance). Sign convention: **DM > 0 ⇒ model beats HAR; DM < 0 ⇒ model worse than HAR.**

---

## 3. Results

### 3.1 Data cleaning, trend and seasonality *(core technique #1)*

The cleaning funnel (§2.1) retains ~1.5M economically meaningful IV observations from ~3.3M raw rows and produces 2,633 arbitrage-free daily surfaces. **Trend / non-stationarity** is addressed formally in §3.3 (ADF/KPSS): the parameter *levels* are persistent (near-unit-root), so all autoregressive modelling is performed on the stationary **first differences**, and the slow-moving long-memory component is handled by the HAR aggregation (§3.6). **Seasonality:** daily SSVI parameters and log-RV exhibit no material deterministic calendar (e.g. day-of-week) seasonality at this frequency — the dominant low-frequency feature is long-memory persistence, not a periodic component. Where a standard technique is genuinely unsuitable for the data we justify its omission rather than forcing it; accordingly we treat a seasonal decomposition as inappropriate for this dataset and proceed with the persistence/long-memory framing instead.

### 3.2 Panel regression, fixed/random effects and hypothesis testing *(core techniques #2, #3)* — Notebook A

We regress Black-76 IV on option structural features (forward moneyness $k$, TTE, a bid-ask liquidity measure) across **~1.49 million** observations over 2,768 trading days, climbing a fixed-effects ladder. The within-transformation (time-demeaning) avoids an explicit ~2,768-column dummy matrix (≈31 GB in double precision).

| Specification | Fixed effects | R² |
|--------------|---------------|----|
| Pooled OLS | none | ~0.34 |
| Day FE | trading day | ~0.80 |
| Day + Maturity FE | day × maturity bucket | **0.86** |
| Surface-Cell FE | day × maturity × moneyness | **0.96** |

**Hypothesis testing.** Forward moneyness and TTE are the dominant, highly significant structural drivers of the IV cross-section. The **Hausman test** rejects random effects (p < 0.001) — fixed effects are the correct specification. The **RESET test** flags mild nonlinearity, which the log(IV) specification reduces. Gauss-Markov diagnostics reveal heteroskedasticity (expected with 1.5M quotes), so HAC standard errors are used throughout; the Jarque-Bera rejection of normality is practically immaterial at this sample size (verified via QQ plot). *Reading:* the surface's **shape** is almost fully explained by moneyness, maturity and day effects — SSVI supplies the geometry, not, as §3.5 shows, an incremental forecasting signal.

### 3.3 Non-stationarity and cointegration *(core technique #7)* — Notebook B

**Stationarity.** Each of the five parameter levels gives a *conflicting* ADF (rejects unit root) vs KPSS (rejects stationarity) verdict → classified "uncertain / borderline I(1)"; all five first differences are cleanly I(0). This justifies differencing before ARMA/VAR.

**Cointegration (α, η).** Motivated by the stochastic-volatility coupling of variance level and vol-of-vol, we test the (α, η) pair. The full-sample **Johansen** trace test returns **rank = 2** and **Engle-Granger** gives p = **0.0002\*\*\***. Crucially, in a *bivariate* system rank = 2 is *full rank* — i.e. evidence that **both series are individually (near-)stationary**, not that they are two I(1) series tied by a single long-run attractor (which would instead appear as rank = 1). The VECM is therefore weakly identified OOS: α $R^2$ = +0.004 (barely positive), η $R^2$ = −0.018 (negative).

**Rolling Johansen (500-day windows).** The full-sample reading is itself **regime-dependent**: rank = 0 in 2.8% of windows, **rank = 1 (genuine single cointegrating vector) in 30.0%**, and rank = 2 (full rank) in **67.2%**; the estimated rank tracks the VIX level (EG calm p = 0.0004, EG stress p = 0.155). The α–η relationship is best read as a *structural, regime-conditional* feature of the SSVI parametrisation rather than a stable tradeable spread.

### 3.4 ARMA, ARMAX and VAR modelling *(core techniques #4, #6)* — Notebook B

**Data window.** 2,632 obs; train 2,105 / test 527 (chronological split).

**ARMA / ARMAX.** BIC over ARMA(p,q), $p,q\in\{0,1,2,3\}$, selects α→ARMA(1,0) and β/ρ/η/γ→ARMA(1,1). Across **all 10 series** (5 levels + 5 differences) the OOS **MSE-ratio = 1.0000** and **$R^2_\text{OOS}$ = 0.0000**: *no* ARMA model beats the random walk. Adding exogenous VIX (ARMAX) does not help — despite in-sample Granger significance of Δlog(VIX) for α and β (p ≈ 0.017–0.028), the signal does not survive OOS. This is the fingerprint of an efficiently-priced surface at the single-lag level. *(A long-memory HAR extension does recover predictable structure in the differenced parameters; see §3.6.)*

**VAR(2) + VECM (technique #6).** A BIC-optimal VAR(2) on the five-parameter system likewise fails to beat the random walk for four of five parameters; **only γ** attains a marginal OOS gain (MSE-ratio = **0.938**). Cross-parameter dynamics add essentially nothing at h = 1, consistent with the ARMA result.

### 3.5 Point and probabilistic forecasting *(core technique #5)* — Notebook C

**Design goal (the first original contribution).** This is the project's primary forecasting deliverable, and its design is deliberately asset-agnostic. VIX-type regime information is specific to the S&P 500 (the VIX is built from SPX options), so we ask whether a single underlying's *own* option surface — through SSVI levels, ATM IV, skew, and nonlinear stress interactions — can supply the same regime signal in a form **portable to any asset with a liquid option market**. The (non-portable) VIX-augmented HAR is the benchmark to beat; **F3_ATM** (HAR + log σ_ATM, requiring only the asset's own options) is the leading *portable* candidate; the M-series progressively adds nonlinear, option-implied stress features.

**Setup.** Target $y_t^{(h)} = \tfrac{1}{h}\sum_{i=1}^h \log\text{RV}_{t+i}$ for $h\in\{1,5,20\}$; train ~2,135 / test ~534 (the test window includes the COVID-19 shock, VIX ≈ 83). A key step (Section 6a) is collinearity control: uncentered λ-interactions reach **VIF = 3,738**, resolved by centering $\tilde\lambda_t=\lambda_t-\bar\lambda_\text{train}$; OLS significance then prunes the feature set. Single-split OOS performance:

| Model | Feat. | h=1 R²_OOS | h=5 R²_OOS | DM(h=5) | h=20 R²_OOS | DM(h=20) |
|-------|-------|------------|------------|---------|-------------|----------|
| Naive | RW | 0.000 | 0.000 | — | 0.000 | — |
| **HAR** | 3 | **0.466** | **0.846** | — | **0.857** | — |
| HAR+VIX | 4 | 0.488 | 0.832 | −0.58 | 0.864 | +0.52 |
| F3_ATM | 4 | 0.465 | 0.840 | −2.47** | 0.846 | −1.87* |
| HAR_rhoJ | 4 | 0.465 | 0.846 | −0.72 | 0.857 | −0.38 |
| M1 | 4 | 0.466 | 0.842 | −2.51** | 0.849 | −1.93* |
| M4_smooth | 9 | 0.449 | 0.819 | −3.59*** | 0.806 | −2.78*** |
| M4+int | 12 | 0.451 | 0.825 | −2.82*** | 0.817 | −2.13** |

**Outcome.** HAR is the best model: every option-augmented specification is statistically *worse* OOS (DM up to −3.59\*\*\* at h=5), and even the portable F3_ATM trails HAR (−2.47\*\* at h=5). In an **expanding-window** re-evaluation (2016–2020 OOS) all option models converge to HAR performance and HAR+VIX is only nominally best (DM not significant, p ≈ 0.26). The richer models overfit the pre-COVID distribution; HAR's parsimony delivers robustness. The honest, well-identified verdict: a single asset's own option surface carries **no incremental RV-forecasting power beyond HAR** through this stressed window — the portable approach is viable (F3_ATM converges to HAR in calm windows) but not superior on this sample.

**Probabilistic forecasting.** Complementing the point forecasts: AR(1)-GARCH(1,1) (Bollerslev 1986) with QMLE errors produces calibrated **80% prediction intervals** for the SSVI parameter paths (Notebook B), and the risk engine below is evaluated by its **ES₉₅** tail coverage (Notebook D).

### 3.6 Original contributions beyond the core requirements

The two main extras are the *portable RV-forecasting study* of §3.5 (Notebook C) and the *market-making risk engine* below (Notebook D); two shorter findings follow.

#### (a) A surface market-making risk engine extending Avellaneda-Stoikov — Notebook D

We extend the single-instrument optimal market-making model of **Avellaneda & Stoikov (2008)** to the **full implied-volatility surface** (a 45-point grid, 9 strikes × 5 maturities, 2,633 days). The Avellaneda-Stoikov logic sets a quoting half-spread around a reservation price that scales with the instrument's volatility and the maker's inventory risk; applied surface-wide this gives a vol-scaled baseline spread `spread_AS`. The problem is that this baseline systematically *under-covers* the realised next-day surface move (it ignores the part of surface risk not captured by a simple vol scaling), so we add an **SSVI-based residual-risk add-on** fed by a HAR-J forecast of surface realized volatility, and calibrate it to a target tail (ES₉₅) coverage by maturity bucket.

- **Add-on term structure.** The calibrated add-on coefficient follows $c^*(T) = 4.950 - 3.208\sqrt{T}$ (R² = **0.949**) — strictly decreasing in maturity, economically sensible: short-dated IV moves are more volatile relative to their long-run expectation and require a larger relative buffer. The √T form is consistent with Brownian surface dynamics.
- **Coverage.** The AS baseline alone covers only **57.3%** of next-day surface moves globally (19.4% at 1M, 48.1% at 3M) — far below the 95% target. With the residual-risk add-on (which makes up **74%** of the total spread) coverage reaches **95.9%** globally (97.4% 1M, 96.9% 3M); pointwise surface containment rises from **47.5%** to **82.9%**.
- **Jump robustness.** A bipower-variation jump filter (BPV) flags 29.7% of days — too loose — so a q95+2σ threshold (1.4% jump rate) is selected; the HAR-J jump term does not improve the surface-RV forecast (DM p = 0.290), so jumps are retained only as a robustness check, not a driver.
- **Sensitivity.** Varying the baseline tightness γ shows a sharp non-linear transition between γ = 1.0 and γ = 2.0 (where `c*` collapses ≈2.2 → ≈0.07): once the baseline quote is wide enough the vol-scaled AS spread alone nears the 95% target and the add-on becomes redundant — identifying the operating regime in which the SSVI add-on actually adds value.

The engine is **portable** (it uses only the asset's own calibrated surface) and turns the SSVI model into a deployable quoting-and-risk tool — the project's most direct operational contribution.

#### (b) Other findings

- **HAR recovers structure in the SSVI parameters.** Where ARMA/ARMAX/VAR all match the random walk (§3.4), a HAR(1,5,22) on the *differenced* parameters beats both the random walk and ARMA for all five series (R²_OOS = 0.51–0.67; strongest for γ at 0.666). Echoing Corsi (2009) and Andrès et al. (2025), the parameter changes are not structureless — their predictability is long-memory and spread across many lags (so a low-order AR misses it), which the HAR's weekly/monthly aggregates capture.

  | Series | HAR MSE-ratio | HAR R²_OOS | ARMA MSE-ratio |
  |--------|---------------|------------|----------------|
  | d_alpha | 0.4336 | 0.5664 | 1.0000 |
  | d_beta  | 0.4494 | 0.5506 | 1.0000 |
  | d_rho   | 0.4908 | 0.5092 | 1.0000 |
  | d_eta   | 0.4358 | 0.5642 | 1.0000 |
  | d_gamma | 0.3339 | **0.6661** | 1.0000 |

- **PCA / regimes (brief).** PCA on surface *changes* gives PC1 = 92.7% (common shift), and a HAR on the PC scores returns negative R²_OOS (−0.004/−0.020/−0.008), consistent with the efficient-surface reading. A 3-state HMM on ΔIV identifies low/mid/high-vol regimes (24%/25%/50% of days); supplementary notebook S2 adds Chow break tests (Volmageddon 2018, COVID 2020), CUSUM stability and Granger-causality checks.

### 3.7 Limitations and robustness

1. **Distributional shift.** The 2018–2020 test window contains COVID-19, an extreme OOS regime; the negative option-feature results should be re-checked on a calmer window (e.g. 2015–2018).
2. **Single underlying / single SSVI variant.** Results are specific to SPX and to the power-law φ(θ); raw SVI, eSSVI or SABR could differ.
3. **VECM / cointegration instability.** Rolling Johansen shows the rank is regime-conditional; the VECM's OOS failure may reflect structural breaks rather than absence of a relationship.
4. **Expanding-window convergence.** Option-augmented models converge to HAR in the expanding window, so part of the single-split deficit is small-sample, not structural.
5. *Minor:* Notebook B's calendar-date axis follows a business-day convention that overstates the span (labels read to 2025); it is purely cosmetic — every statistic is defined on the row index, not the date — and the canonical 2010–2020 mapping (Notebook D / engine) is the correct one.

---

## 4. Conclusion

The project's central empirical result is **negative with informational content**: a single asset's structurally rich, arbitrage-free SSVI surface does *not* improve on a parsimonious HAR model for realized-volatility forecasting (every option-augmented model is significantly worse OOS, DM up to −3.59\*\*\*), and at the daily frequency the surface behaves like a near-efficiently-priced, near-martingale object — ARMA, ARMAX and VAR all match the random walk. This directly answers our first research aim: the asset-agnostic ambition of replacing the SPX-specific VIX with portable, option-implied regime features is *viable but not superior* on this sample — the leading portable model (F3_ATM) converges to HAR in calm windows but cannot beat it through the COVID-19 shock. That HAR wins is itself the economically meaningful finding: backward-looking long-memory structure, not forward-looking surface geometry, is what carries short-horizon RV predictability here.

The second aim delivers the project's main operational contribution: extending **Avellaneda & Stoikov (2008)** to the full implied-volatility surface, an SSVI-based residual-risk add-on with a √T-calibrated term structure (R² = 0.949) lifts ES₉₅ coverage from **57.3%** (baseline) to **95.9%** — a portable, deployable surface-quoting-and-risk tool that uses only the asset's own options. Around these two contributions sit three supporting results: a fully arbitrage-free SSVI calibration database (2,633 daily surfaces, 95.1% success); a panel decomposition showing 86% of IV cross-sectional variation (96% with Surface-Cell FE) is explained by moneyness, maturity and day effects — SSVI supplies shape, not signal; and the one positive predictability finding, that the *differenced* SSVI parameters carry long-memory (HAR) structure (R²_OOS up to 0.67) invisible to ARMA. Every core econometric technique is exercised on the data, and the two extensions place the standard toolbox inside a genuine quantitative-finance application.

---

## 5. Bibliography

- Andersen, T., Bollerslev, T., Diebold, F. & Labys, P. (2003). Modeling and forecasting realized volatility. *Econometrica*, 71(2), 579–625.
- Andrès, H., Boumezoued, A. & Jourdain, B. (2025). The implied volatility surface (also) is path-dependent. *arXiv preprint*.
- Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3), 217–224.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307–327.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174–196.
- Diebold, F. & Mariano, R. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Engle, R. & Granger, C. (1987). Co-integration and error correction: representation, estimation, and testing. *Econometrica*, 55(2), 251–276.
- Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance*, 14(1), 59–71.
- Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. *Econometrica*, 59(6), 1551–1580.
