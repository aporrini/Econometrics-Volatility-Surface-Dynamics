"""
ssvi_mm_risk_engine.py
======================
Helper library for notebook 07 — SSVI Surface Risk Engine.

All functions are pure / stateless unless a Path argument is provided for I/O.
No code is executed at import time.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.stats import norm
import statsmodels.api as smapi
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════
# 1. LOADING
# ══════════════════════════════════════════════════════════════════════════

def find_ssvi_results_file(output_dir: Path) -> Path:
    """Return the most recently modified SSVI calibration CSV in *output_dir*."""
    cands = sorted(
        list(output_dir.glob('ssvi_all_dates*results*.csv')) +
        list(output_dir.glob('ssvi_all_dates*.csv')),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(f'No SSVI results CSV found in {output_dir}')
    return cands[0]


def clean_ssvi_results(params_raw: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where success=True; reset integer index."""
    return params_raw[params_raw['success'] == True].copy().reset_index(drop=True)


_SSVI_START: pd.Timestamp = pd.Timestamp('2010-01-04')


def load_ssvi_results(
    data_dir: Path,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Load SSVI calibration results and set a DatetimeIndex.

    Parameters
    ----------
    data_dir : Path
        Directory containing the SSVI calibration CSV
        (``ssvi_all_dates_clean_results.csv`` or similar).
    output_dir : Path | None
        When provided, a cached IV panel (``ssvi_surface_iv_panel.csv``) is
        checked as a secondary source of dates.  If the panel exists and its
        length matches, its index is used instead of the computed dates.
        This argument is kept for backward-compatibility; it is not required.

    Returns
    -------
    pd.DataFrame with DatetimeIndex, one row per successfully calibrated date.
    """
    path      = find_ssvi_results_file(data_dir)
    print(f'Loading SSVI results from: {path.name}')
    params_ok = clean_ssvi_results(pd.read_csv(path))

    # Primary: derive dates from time_elapsed
    # time_elapsed is the number of calendar days since 2010-01-04
    if 'time_elapsed' in params_ok.columns:
        dates = _SSVI_START + pd.to_timedelta(params_ok['time_elapsed'], unit='D')
        params_ok.index = pd.DatetimeIndex(dates)
        params_ok.index.name = None

    # Secondary: validate / override with IV panel if available and matching
    if output_dir is not None:
        iv_src = output_dir / 'ssvi_surface_iv_panel.csv'
        if iv_src.exists():
            iv_ref = pd.read_csv(iv_src, index_col=0, parse_dates=True)
            if len(iv_ref) == len(params_ok):
                params_ok.index = iv_ref.index

    return params_ok


# ══════════════════════════════════════════════════════════════════════════
# 2. SURFACE RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════

def get_grid_metadata(
    k_grid: np.ndarray,
    t_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], Dict[str, str]]:
    """Expand k×T meshgrid; return (k_flat, t_flat, grid_cols, t_labels, mat_t_map)."""
    KK, TT    = np.meshgrid(k_grid, t_grid)
    k_flat    = KK.ravel()
    t_flat    = TT.ravel()
    grid_cols = [f'iv_k_{k:.2f}_T_{t:.2f}' for t, k in zip(t_flat, k_flat)]
    t_labels  = ['1M', '3M', '6M', '1Y', '2Y']
    mat_t_map = {lbl: f'{t:.2f}' for lbl, t in zip(t_labels, t_grid)}
    return k_flat, t_flat, grid_cols, t_labels, mat_t_map


def parse_grid_column(col: str) -> Tuple[float, float]:
    """Parse 'iv_k_-0.10_T_0.25' → (k=-0.10, T=0.25)."""
    k = float(col.split('_k_')[1].split('_T_')[0])
    T = float(col.split('_T_')[1])
    return k, T


def build_iv_panel(
    params_ok: pd.DataFrame,
    k_flat: np.ndarray,
    t_flat: np.ndarray,
    grid_cols: List[str],
    src_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load IV panel from *src_path* if it exists; otherwise reconstruct from SSVI params.
    Saves result to *out_path* when provided.
    """
    if src_path is not None and src_path.exists():
        iv_panel = pd.read_csv(src_path, index_col=0, parse_dates=True).sort_index()
    else:
        _p_cols = [c for c in ['alpha', 'beta', 'rho', 'eta', 'gamma'] if c in params_ok.columns]
        recs = []
        for date, row in params_ok[_p_cols].iterrows():
            if row.isnull().any():
                continue
            a, b, r, e, g = (float(row[c]) for c in ['alpha', 'beta', 'rho', 'eta', 'gamma'])
            iv_vals = []
            for k, T in zip(k_flat, t_flat):
                theta = np.exp(a) * (T ** b)
                phi   = e * theta ** (-g) / (1.0 + e * theta ** (1.0 - g))
                w     = (theta / 2.0) * (1.0 + r * phi * k + np.sqrt((phi * k + r) ** 2 + 1.0 - r ** 2))
                iv_vals.append(float(np.sqrt(max(w, 0.0) / T)) if T > 0 else np.nan)
            recs.append(pd.Series(iv_vals, index=grid_cols, name=date))
        iv_panel = pd.DataFrame(recs).dropna(how='all')

    if out_path is not None:
        iv_panel.to_csv(out_path)
    return iv_panel


# ══════════════════════════════════════════════════════════════════════════
# 3. DELTA IV AND REALIZED VOLATILITY
# ══════════════════════════════════════════════════════════════════════════

def compute_delta_iv_panel(iv_panel: pd.DataFrame) -> pd.DataFrame:
    """Daily first differences of the IV panel; first row dropped."""
    return iv_panel.diff().dropna()


def compute_surface_move(delta_iv: pd.DataFrame) -> pd.Series:
    """Equal-weighted RMS of ΔIV across all grid columns (global surface move).

    Missing ΔIV values are skipped (not treated as zero moves).
    Emits a warning for any day where fewer than 80% of grid points are valid.
    """
    s = np.sqrt(delta_iv.pow(2).mean(axis=1, skipna=True))
    s.name = 'surface_move'
    valid_ratio = delta_iv.notna().mean(axis=1)
    low = valid_ratio[valid_ratio < 0.80]
    if len(low) > 0:
        warnings.warn(
            f'{len(low)} days have <80% valid ΔIV grid points '
            f'(min={low.min():.1%}). RMS computed on available points.',
            stacklevel=2,
        )
    return s


def compute_move_by_maturity(
    delta_iv: pd.DataFrame,
    grid_cols: List[str],
    t_labels: List[str],
    mat_t_map: Dict[str, str],
) -> pd.DataFrame:
    """Per-maturity RMS of ΔIV. Returns DataFrame with columns move_1M … move_2Y."""
    move = pd.DataFrame(index=delta_iv.index)
    for t_lbl, t_str in mat_t_map.items():
        cols_t = [c for c in grid_cols if f'_T_{t_str}' in c]
        move[f'move_{t_lbl}'] = np.sqrt(delta_iv[cols_t].pow(2).mean(axis=1, skipna=True))
    return move


def build_future_rv_targets(
    smove: pd.Series,
    move_by_mat: pd.DataFrame,
    horizons: Sequence[int],
    t_labels: List[str],
) -> pd.DataFrame:
    """
    future_rv_h(t) = sqrt( mean_{j=1}^h smove(t+j)^2 ).
    Uses .rolling(h).mean().shift(-h) — no leakage.
    """
    targets = pd.DataFrame(index=smove.index)
    sm_sq   = smove.pow(2)
    for h in horizons:
        targets[f'future_rv_{h}d'] = np.sqrt(
            sm_sq.rolling(h, min_periods=h).mean().shift(-h)
        )
        for t_lbl in t_labels:
            col = f'move_{t_lbl}'
            if col in move_by_mat.columns:
                targets[f'future_rv_{h}d_{t_lbl}'] = np.sqrt(
                    move_by_mat[col].pow(2).rolling(h, min_periods=h).mean().shift(-h)
                )
    return targets


# ══════════════════════════════════════════════════════════════════════════
# 4. JUMP DETECTION
# ══════════════════════════════════════════════════════════════════════════

def detect_jumps_bpv(smove: pd.Series) -> pd.Series:
    """BPV jump flag (Barndorff-Nielsen & Shephard 2004). Returns binary pd.Series."""
    sm_sq  = smove.pow(2)
    bpv_sq = (np.pi / 2) * smove.abs() * smove.abs().shift(1)
    j_sq   = np.maximum(sm_sq - bpv_sq, 0.0)
    return (j_sq > 0).astype(float)


def detect_jumps_rolling_q95_plus_sigma(
    smove: pd.Series,
    window: int = 252,
    min_periods: int = 50,
) -> pd.Series:
    """Jump flag: smove > rolling_q95 + 2 × rolling_std."""
    rq = smove.rolling(window, min_periods=min_periods).quantile(0.95)
    rs = smove.rolling(window, min_periods=min_periods).std()
    return ((smove > (rq + 2 * rs)).astype(float)).fillna(0.0)


def detect_jumps_rolling_q99(
    smove: pd.Series,
    window: int = 252,
    min_periods: int = 50,
) -> pd.Series:
    """Jump flag: smove > rolling_q99."""
    rq = smove.rolling(window, min_periods=min_periods).quantile(0.99)
    return ((smove > rq).astype(float)).fillna(0.0)


def choose_jump_method(
    smove: pd.Series,
    target_lo: float = 5.0,
    target_hi: float = 15.0,
    target_mid: float = 10.0,
) -> Tuple[pd.Series, str, float, Dict]:
    """
    Auto-select between Method A (q95+2σ) and Method B (q99).
    Prefers the one in [target_lo%, target_hi%], closest to target_mid%.
    Returns (j_flag_selected, selected_key, selected_pct, candidates_dict).
    candidates_dict = {'A': (flag_A, pct_A), 'B': (flag_B, pct_B)}.
    """
    flag_A = detect_jumps_rolling_q95_plus_sigma(smove)
    flag_B = detect_jumps_rolling_q99(smove)
    pct_A  = float(flag_A.mean() * 100)
    pct_B  = float(flag_B.mean() * 100)
    candidates  = {'A': (flag_A, pct_A), 'B': (flag_B, pct_B)}
    in_range    = {k: v for k, v in candidates.items() if target_lo <= v[1] <= target_hi}
    pool        = in_range if in_range else candidates
    key         = min(pool, key=lambda k: abs(pool[k][1] - target_mid))
    return candidates[key][0], key, candidates[key][1], candidates


# ══════════════════════════════════════════════════════════════════════════
# 5. FORECASTING
# ══════════════════════════════════════════════════════════════════════════

def qlike_loss(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """QLIKE: E[ratio - log(ratio) - 1] where ratio = exp(y_true - y_pred)."""
    ratio = np.exp(np.asarray(y_true_log) - np.asarray(y_pred_log))
    return float(np.nanmean(ratio - np.log(np.maximum(ratio, 1e-12)) - 1))


def diebold_mariano_test(
    e1: pd.Series,
    e2: pd.Series,
    h: int = 5,
) -> Tuple[float, float]:
    """
    Diebold-Mariano (1995) with Newey-West HAC variance.
    H0: equal MSE. Positive stat → e1 has larger MSE (model 2 better).
    Returns (stat, two-sided p-value).
    """
    d     = e1 ** 2 - e2 ** 2
    d     = d.dropna()
    n     = len(d)
    if n < 20:
        return np.nan, np.nan
    d_bar  = d.mean()
    gamma0 = ((d - d_bar) ** 2).mean()
    gammas = [((d - d_bar).iloc[k:] * (d - d_bar).iloc[:-k]).mean()
              for k in range(1, min(h, n // 4))]
    lrv = gamma0 + 2 * sum(gammas)
    if lrv <= 0:
        return np.nan, np.nan
    stat = d_bar / np.sqrt(max(lrv / n, 1e-15))
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def make_har_features(
    smove: pd.Series,
    j_flag: Optional[pd.Series] = None,
) -> Dict[str, pd.Series]:
    """
    Compute HAR feature set from a univariate volatility series.
    When j_flag is provided, also computes jump-decomposition features
    (j_cnt22, log_j22, log_c5, log_j5).
    Returns dict of named pd.Series.
    """
    sm_sq    = smove.pow(2)
    features: Dict[str, pd.Series] = {
        'log_rv1':  np.log(smove + 1e-8),
        'log_rv5':  np.log(np.sqrt(sm_sq.rolling(5,  min_periods=3).mean())  + 1e-8),
        'log_rv22': np.log(np.sqrt(sm_sq.rolling(22, min_periods=10).mean()) + 1e-8),
        'log_ewma': np.log(np.sqrt(sm_sq.ewm(alpha=0.06, min_periods=5).mean()) + 1e-8),
    }
    if j_flag is not None:
        j_sq_comp = j_flag * sm_sq
        bpv_comp  = np.maximum(sm_sq - j_sq_comp, 0.0)
        features.update({
            'j_cnt22': j_flag.rolling(22, min_periods=1).sum(),
            'log_j22': np.log(np.sqrt(j_sq_comp.rolling(22, min_periods=1).mean()) + 1e-8),
            'log_c5':  np.log(np.sqrt(bpv_comp.rolling(5,  min_periods=3).mean())  + 1e-8),
            'log_j5':  np.log(np.sqrt(j_sq_comp.rolling(5, min_periods=3).mean())  + 1e-8),
        })
    return features


def fit_har_model(
    df_all: pd.DataFrame,
    feat_cols: List[str],
    tgt_col: str,
    tr_idx: pd.Index,
    pred_idx: pd.Index,
) -> pd.Series:
    """Fit OLS on *tr_idx*, predict on *pred_idx*. Returns log-scale pd.Series."""
    sub = df_all[[tgt_col] + feat_cols].dropna()
    tr  = sub.index.intersection(tr_idx)
    if len(tr) < 30:
        return pd.Series(dtype=float)
    X_tr = smapi.add_constant(sub.loc[tr, feat_cols], has_constant='add')
    y_tr = sub.loc[tr, tgt_col]
    ols  = smapi.OLS(y_tr, X_tr).fit()
    pr   = sub.index.intersection(pred_idx)
    if len(pr) == 0:
        return pd.Series(dtype=float)
    X_pr = smapi.add_constant(sub.loc[pr, feat_cols], has_constant='add')
    X_pr = X_pr.reindex(columns=X_tr.columns, fill_value=0.0)
    return pd.Series(ols.predict(X_pr).values, index=pr)


def compute_forecast_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
) -> Dict:
    """Return dict {n, MAE_log, RMSE_log, QLIKE} for log-scale arrays."""
    yt = np.asarray(y_true_log, dtype=float)
    yp = np.asarray(y_pred_log, dtype=float)
    ok = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[ok], yp[ok]
    return {
        'n':        int(len(yt)),
        'MAE_log':  float(np.mean(np.abs(yt - yp))),
        'RMSE_log': float(np.sqrt(np.mean((yt - yp) ** 2))),
        'QLIKE':    qlike_loss(yt, yp),
    }


# ══════════════════════════════════════════════════════════════════════════
# 6. PCA
# ══════════════════════════════════════════════════════════════════════════

def fit_train_pca(
    delta_iv: pd.DataFrame,
    split_train: pd.Index,
    n_components: int = 10,
) -> Tuple[StandardScaler, PCA]:
    """Fit StandardScaler + PCA on the train set only. Returns (scaler, pca)."""
    X_tr   = delta_iv.reindex(split_train).fillna(0.0)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_tr)
    n_comp = min(n_components, X_sc.shape[1], X_sc.shape[0] - 1)
    pca    = PCA(n_components=n_comp, random_state=42)
    pca.fit(X_sc)
    return scaler, pca


def transform_pca_splits(
    delta_iv: pd.DataFrame,
    scaler: StandardScaler,
    pca: PCA,
    split_train: pd.Index,
    split_val: pd.Index,
    split_test: pd.Index,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Project train/val/test through the fitted scaler + PCA.
    Returns (pc_scores_df, loadings_df, explained_variance_ratio).
    """
    n_comp   = pca.n_components_
    pc_names = [f'PC{i+1}' for i in range(n_comp)]
    parts    = [
        (split_train, delta_iv.reindex(split_train).fillna(0.0)),
        (split_val,   delta_iv.reindex(split_val).fillna(0.0)),
        (split_test,  delta_iv.reindex(split_test).fillna(0.0)),
    ]
    all_scores, all_idx = [], []
    for idx, X in parts:
        all_scores.append(pca.transform(scaler.transform(X)))
        all_idx.extend(idx.tolist())

    pc_df = pd.DataFrame(
        np.vstack(all_scores), index=all_idx, columns=pc_names
    ).sort_index()
    loadings_df = pd.DataFrame(
        pca.components_.T, index=delta_iv.columns.tolist(), columns=pc_names
    )
    return pc_df, loadings_df, pca.explained_variance_ratio_


def save_pca_outputs(
    pc_df: pd.DataFrame,
    loadings_df: pd.DataFrame,
    output_dir: Path,
    prefix: str = '07',
) -> None:
    """Save PC scores and loadings CSVs."""
    pc_df.to_csv(output_dir / f'{prefix}_pca_delta_iv_scores.csv')
    loadings_df.to_csv(output_dir / f'{prefix}_pca_delta_iv_loadings.csv')


# ══════════════════════════════════════════════════════════════════════════
# 7. HMM
# ══════════════════════════════════════════════════════════════════════════

def fit_hmm_regimes(
    log_rv22: pd.Series,
    split_train: pd.Index,
    split_val: pd.Index,
    n_states: int = 3,
) -> Tuple[GaussianHMM, pd.Series, Dict[int, int], pd.DataFrame]:
    """
    Fit Gaussian HMM on train+val; apply Viterbi on the full series.
    Regimes sorted 0=low-vol … n-1=high-vol by mean log RV.
    Returns (hmm, regime_series, remap_dict, transition_df).
    ⚠ Viterbi path is retrospective — descriptive use only.
    """
    fit_dates = split_train.append(split_val)
    fit_data  = log_rv22.reindex(fit_dates).dropna().values.reshape(-1, 1)
    hmm = GaussianHMM(n_components=n_states, covariance_type='full',
                      n_iter=200, random_state=42, tol=1e-4)
    hmm.fit(fit_data)

    full = log_rv22.dropna()
    raw  = pd.Series(hmm.predict(full.values.reshape(-1, 1)), index=full.index)
    means = {r: full[raw == r].mean() for r in range(n_states)}
    remap = {old: new for new, old in enumerate(sorted(means, key=lambda r: means[r]))}
    regime_s = raw.map(remap).rename('regime')

    trans_orig  = hmm.transmat_
    trans_relab = np.zeros_like(trans_orig)
    for old_i, new_i in remap.items():
        for old_j, new_j in remap.items():
            trans_relab[new_i, new_j] = trans_orig[old_i, old_j]
    lbl = [f'R{r}' for r in range(n_states)]
    trans_df = pd.DataFrame(trans_relab, index=lbl, columns=lbl)
    return hmm, regime_s, remap, trans_df


def summarize_hmm_regimes(
    regime_s: pd.Series,
    log_rv22: pd.Series,
    regime_labels: Dict[int, str],
) -> pd.DataFrame:
    """Return summary DataFrame (regime, label, n, pct, mean_rv)."""
    # Convert to numpy to avoid pandas boolean-indexer alignment errors
    rv_arr  = log_rv22.reindex(regime_s.index).to_numpy(dtype=float)
    reg_arr = regime_s.to_numpy()
    n_total = len(reg_arr)
    rows = []
    for r, lbl in regime_labels.items():
        mask   = reg_arr == r
        n_r    = int(mask.sum())
        mean_rv = float(np.exp(rv_arr[mask].mean())) if n_r > 0 else np.nan
        rows.append({'regime': r, 'label': lbl, 'n': n_r,
                     'pct': n_r / n_total * 100, 'mean_rv': mean_rv})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# 8. SPREAD ENGINE
# ══════════════════════════════════════════════════════════════════════════

def compute_avellaneda_style_spread(
    iv_panel: pd.DataFrame,
    grid_cols: List[str],
    k_flat: np.ndarray,
    mat_t_map: Dict[str, str],
    gamma_as: float = 0.5,
    kappa_k: float = 0.5,
    dt: float = 1 / 252,
) -> pd.DataFrame:
    """
    AS-style proxy: spread_AS(k,T,t) = γ × σ_ATM(T,t) × √dt × (1 + κ|k|).
    This is a heuristic proxy, not the full Avellaneda-Stoikov model.
    """
    spread = pd.DataFrame(np.nan, index=iv_panel.index, columns=grid_cols)
    for t_str in set(mat_t_map.values()):
        atm_col  = f'iv_k_0.00_T_{t_str}'
        mat_cols = [c for c in grid_cols if f'_T_{t_str}' in c]
        if atm_col not in iv_panel.columns or not mat_cols:
            continue
        sigma_atm = iv_panel[atm_col].ffill()
        for col in mat_cols:
            k_val = float(col.split('_k_')[1].split('_T_')[0])
            spread[col] = gamma_as * sigma_atm * np.sqrt(dt) * (1.0 + kappa_k * abs(k_val))
    return spread


def expected_shortfall(values: np.ndarray, alpha: float = 0.95) -> float:
    """ES_alpha: mean of obs at or above the alpha-quantile. Returns NaN if < 10 obs."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals >= 0)]
    if len(vals) < 10:
        return np.nan
    q    = np.quantile(vals, alpha)
    tail = vals[vals >= q]
    return float(tail.mean()) if len(tail) > 0 else np.nan


def calibrate_c_es95(
    split_val: pd.Index,
    targets_df: pd.DataFrame,
    calib_targets: Dict,
    fc_val: Dict,
    log_ewma: pd.Series,
    calib_alpha: float = 0.95,
) -> pd.DataFrame:
    """
    Calibrate c*(bucket) = ES_{calib_alpha}( max(actual_rv - spread_AS, 0) / RV_hat )
    using the validation set only.

    calib_targets: {bucket_key: (tgt_col_name, spread_as_series)}
    fc_val:        {'HAR-J|bucket': log-scale val forecast pd.Series}
    Returns DataFrame indexed by bucket, columns [c_star, n_val].
    """
    rows = []
    for bucket_key, (tgt_col, sp_as_ser) in calib_targets.items():
        actual_rv = targets_df[tgt_col].reindex(split_val).dropna()
        fc_key    = f'HAR-J|{"global" if bucket_key == "global" else bucket_key}'
        fc_log    = fc_val.get(fc_key, pd.Series(dtype=float))
        if len(fc_log) == 0:
            fc_log = log_ewma
        fc_rv  = np.exp(fc_log.reindex(split_val).clip(upper=5)).clip(lower=1e-8)
        sp_va  = sp_as_ser.reindex(split_val).ffill().bfill()
        common = actual_rv.index.intersection(fc_rv.dropna().index)
        if len(common) < 10:
            c_star = 1.5
        else:
            act    = actual_rv.loc[common]
            fc     = fc_rv.loc[common]
            sp     = sp_va.reindex(common)
            excess = np.maximum(act - sp, 0.0)
            ratio  = (excess / (fc + 1e-8)).clip(0, 50)
            c_star = expected_shortfall(ratio.values, calib_alpha)
            if np.isnan(c_star):
                c_star = 1.5
        rows.append({'bucket': bucket_key, 'c_star': c_star, 'n_val': len(common)})
    return pd.DataFrame(rows).set_index('bucket')


def compute_spread_final(
    iv_panel: pd.DataFrame,
    grid_cols: List[str],
    mat_t_map: Dict[str, str],
    spread_AS: pd.DataFrame,
    fc_test: Dict,
    calib_df: pd.DataFrame,
    log_ewma: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build spread_final = spread_AS + c*(bucket) × RV_hat.
    Returns (spread_final, addon_panel).
    """
    def _c(bucket_key: str) -> float:
        if bucket_key in calib_df.index:
            return float(calib_df.loc[bucket_key, 'c_star'])
        return float(calib_df.loc['global', 'c_star']) if 'global' in calib_df.index else 1.5

    spread_final = spread_AS.copy()
    addon_panel  = pd.DataFrame(0.0, index=iv_panel.index, columns=grid_cols)

    for t_lbl, t_str in mat_t_map.items():
        c_curr   = _c(t_lbl)
        fc_key   = f'HAR-J|{t_lbl}' if f'HAR-J|{t_lbl}' in fc_test else 'HAR-J|global'
        fc_log   = fc_test.get(fc_key, pd.Series(dtype=float))
        if len(fc_log) == 0:
            fc_log = log_ewma
        fc_rv    = np.exp(fc_log).clip(lower=1e-8).reindex(iv_panel.index)
        mat_cols = [c for c in grid_cols if f'_T_{t_str}' in c]
        for col in mat_cols:
            valid = fc_rv.notna()
            addon_panel.loc[valid, col]  = c_curr * fc_rv[valid]
            spread_final.loc[valid, col] = (
                spread_AS.loc[valid, col].fillna(0.0) + c_curr * fc_rv[valid]
            )
    return spread_final, addon_panel


def backtest_spread_coverage(
    split_test: pd.Index,
    calib_targets: Dict,
    targets_df: pd.DataFrame,
    spread_final: pd.DataFrame,
    spread_AS: pd.DataFrame,
    grid_cols: Optional[List[str]] = None,
    mat_t_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Coverage backtest on test set. Returns bt_df with one row per bucket."""
    rows = []
    for bucket_key, (tgt_col, _) in calib_targets.items():
        actual_rv = targets_df[tgt_col].reindex(split_test).dropna()
        if bucket_key == 'global':
            sf_ser = spread_final.mean(axis=1).reindex(split_test)
            as_ser = spread_AS.mean(axis=1).reindex(split_test)
        else:
            if mat_t_map is None or grid_cols is None:
                continue
            t_str    = mat_t_map[bucket_key]
            mat_cols = [c for c in grid_cols if f'_T_{t_str}' in c]
            sf_ser   = spread_final[mat_cols].mean(axis=1).reindex(split_test)
            as_ser   = spread_AS[mat_cols].mean(axis=1).reindex(split_test)

        addon_ser = sf_ser - as_ser
        common    = actual_rv.index.intersection(sf_ser.dropna().index)
        if len(common) < 10:
            continue
        act, sf, as_, add = (actual_rv.loc[common], sf_ser.loc[common],
                              as_ser.loc[common],   addon_ser.loc[common])
        cov    = act <= sf
        cov_as = act <= as_
        rows.append({
            'bucket':               bucket_key,
            'n':                    len(common),
            'coverage_AS':          float(cov_as.mean()),
            'coverage':             float(cov.mean()),
            'coverage_improvement': float(cov.mean() - cov_as.mean()),
            'exceedance':           float((~cov).mean()),
            'mean_AS':              float(as_.mean()),
            'mean_addon':           float(add.mean()),
            'mean_final':           float(sf.mean()),
            'addon_share':          float(add.mean() / (sf.mean() + 1e-12)),
            'underquote_loss':      float((act[~cov] - sf[~cov]).mean()) if (~cov).sum() > 0 else 0.0,
            'overquote_proxy':      float((sf[cov]  - act[cov]).mean())  if cov.sum()  > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def compute_vega_weighted_robustness(
    iv_panel: pd.DataFrame,
    spread_final: pd.DataFrame,
    targets_df: pd.DataFrame,
    grid_cols: List[str],
    mat_t_map: Dict[str, str],
    split_test: pd.Index,
    h_main: int = 5,
) -> pd.DataFrame:
    """Vega-weighted coverage robustness check. Returns one row per maturity."""
    rows = []
    for t_lbl, t_str in mat_t_map.items():
        T_val    = float(t_str)
        tgt_col  = f'future_rv_{h_main}d_{t_lbl}'
        mat_cols = [c for c in grid_cols if f'_T_{t_str}' in c]
        atm_col  = f'iv_k_0.00_T_{t_str}'
        if tgt_col not in targets_df.columns or not mat_cols or atm_col not in iv_panel.columns:
            continue
        sigma  = iv_panel[atm_col].clip(lower=0.01).ffill()
        sv     = sigma * np.sqrt(T_val)
        vega_d = {}
        for col in mat_cols:
            k_val = float(col.split('_k_')[1].split('_T_')[0])
            d1    = (-k_val + sv / 2) / sv.clip(lower=1e-8)
            vega_d[col] = norm.pdf(d1) * np.sqrt(T_val)
        vega_mat = pd.DataFrame(vega_d, index=iv_panel.index)

        act_rv = targets_df[tgt_col].reindex(split_test).dropna()
        sf_m   = spread_final[mat_cols].reindex(split_test).mean(axis=1)
        vg_m   = vega_mat[mat_cols].reindex(split_test).mean(axis=1).ffill()
        common = act_rv.index.intersection(sf_m.dropna().index).intersection(vg_m.dropna().index)
        if len(common) < 10:
            continue
        act_v, sf_v, vg_v = act_rv.loc[common], sf_m.loc[common], vg_m.loc[common]
        rows.append({
            'maturity':                t_lbl,
            'mean_vega':               float(vg_v.mean()),
            'coverage_unweighted':     float((act_v <= sf_v).mean()),
            'coverage_vegaweighted':   float((act_v * vg_v <= sf_v * vg_v).mean()),
            'vega_weighted_shortfall': float((np.maximum(act_v - sf_v, 0.0) * vg_v).mean()),
            'n':                       len(common),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# 9. PLOTTING
# ══════════════════════════════════════════════════════════════════════════

def plot_surface_move(smove: pd.Series, plot_dir: Path, prefix: str = '07') -> None:
    """Save plots/{prefix}_surface_move.png."""
    fig, ax = plt.subplots(figsize=(14, 4))
    smove.plot(ax=ax, lw=0.6, color='steelblue', alpha=0.7, label='daily')
    smove.rolling(22, min_periods=10).mean().plot(ax=ax, lw=1.5, color='navy', label='22d MA')
    ax.set_title('Global SSVI Surface Move  (equal-weighted RMS of ΔIV)', fontsize=11)
    ax.set_ylabel('vol-units')
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig(plot_dir / f'{prefix}_surface_move.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_move_by_maturity(move_by_mat: pd.DataFrame, plot_dir: Path, prefix: str = '07') -> None:
    """Save plots/{prefix}_move_by_maturity.png."""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig, ax = plt.subplots(figsize=(14, 4))
    for col, color in zip(move_by_mat.columns, colors):
        lbl = col.replace('move_', '')
        move_by_mat[col].rolling(22, min_periods=10).mean().plot(
            ax=ax, lw=1.3, color=color, label=lbl, alpha=0.85)
    ax.set_title('SSVI Move by Maturity  (22d rolling mean)', fontsize=11)
    ax.set_ylabel('vol-units')
    ax.legend(fontsize=9, ncol=5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig(plot_dir / f'{prefix}_move_by_maturity.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_jump_detection_comparison(
    smove: pd.Series,
    j_flag_old: pd.Series,
    j_flag_new: pd.Series,
    selected_key: str,
    old_pct: float,
    new_pct: float,
    plot_dir: Path,
    prefix: str = '07',
) -> None:
    """Save plots/{prefix}_jump_detection_comparison.png."""
    TARGET_LO, TARGET_HI = 5.0, 15.0
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle('Jump Detection Comparison  (BPV vs rolling-threshold)', fontsize=11, fontweight='bold')

    ax = axes[0]
    smove.plot(ax=ax, lw=0.5, color='grey', alpha=0.6, label='surface_move')
    idx_old, idx_new = j_flag_old.astype(bool), j_flag_new.astype(bool)
    ax.scatter(smove.index[idx_old], smove[idx_old],
               s=10, color='red', alpha=0.5, label=f'BPV ({old_pct:.0f}%)', zorder=3)
    ax.scatter(smove.index[idx_new], smove[idx_new],
               s=18, color='blue', alpha=0.7, marker='^',
               label=f'Method {selected_key} ({new_pct:.0f}%)', zorder=4)
    ax.set_ylabel('surface_move'); ax.legend(fontsize=8, ncol=3)
    ax.set_title('Jump Days: BPV vs Selected Method')

    ax = axes[1]
    j_flag_old.rolling(252, min_periods=50).mean().mul(100).plot(
        ax=ax, lw=1.2, color='red', label='BPV 1Y rate')
    j_flag_new.rolling(252, min_periods=50).mean().mul(100).plot(
        ax=ax, lw=1.2, color='blue', label=f'Method {selected_key} 1Y rate')
    ax.axhline(TARGET_LO, ls=':', color='grey', lw=0.8)
    ax.axhline(TARGET_HI, ls=':', color='grey', lw=0.8)
    ax.set_ylabel('% jump days (1Y window)'); ax.legend(fontsize=8)
    ax.set_title('Rolling Jump Rate')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(plot_dir / f'{prefix}_jump_detection_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_pca_loadings(
    loadings_df: pd.DataFrame,
    k_grid: np.ndarray,
    t_grid: np.ndarray,
    t_labels: List[str],
    ev: np.ndarray,
    plot_dir: Path,
    prefix: str = '07',
) -> None:
    """Save plots/{prefix}_pca_loadings.png."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('PCA Loadings on ΔIV  (PC1=level, PC2=skew, PC3=curvature)',
                 fontsize=11, fontweight='bold')
    for ax, pc_i, title in zip(axes, [1, 2, 3],
                                ['PC1 (level)', 'PC2 (skew)', 'PC3 (curvature)']):
        mat = loadings_df[f'PC{pc_i}'].values.reshape(len(t_grid), len(k_grid))
        lim = np.nanmax(np.abs(mat))
        im  = ax.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim, origin='upper')
        ax.set_xticks(range(len(k_grid)))
        ax.set_xticklabels([f'{k:.2f}' for k in k_grid], rotation=45, fontsize=7)
        ax.set_yticks(range(len(t_grid))); ax.set_yticklabels(t_labels, fontsize=8)
        ax.set_xlabel('k'); ax.set_ylabel('T')
        ax.set_title(f'{title}\n({ev[pc_i-1]*100:.1f}%)', fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(plot_dir / f'{prefix}_pca_loadings.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_pca_scores(pc_df: pd.DataFrame, plot_dir: Path, prefix: str = '07') -> None:
    """Save plots/{prefix}_pca_scores.png."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    fig.suptitle('PC Scores  (daily ΔIV surface projections)', fontsize=10)
    for i, (ax, col, color) in enumerate(zip(axes, ['PC1', 'PC2', 'PC3'],
                                              ['#1f77b4', '#ff7f0e', '#2ca02c'])):
        pc_df[col].plot(ax=ax, lw=0.5, color=color, alpha=0.7)
        pc_df[col].rolling(22).mean().plot(ax=ax, lw=1.2, color='black', label='22d MA')
        ax.set_ylabel(col, fontsize=9); ax.axhline(0, lw=0.5, color='grey', ls='--')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig(plot_dir / f'{prefix}_pca_scores.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_hmm_regimes(
    smove: pd.Series,
    regime_s: pd.Series,
    regime_labels: Dict[int, str],
    plot_dir: Path,
    prefix: str = '07',
) -> None:
    """Save plots/{prefix}_hmm_regimes.png."""
    reg_colors = {0: 'green', 1: 'orange', 2: 'red'}
    fig, axes  = plt.subplots(2, 1, figsize=(14, 6),
                               gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('HMM Regime Detection  (3-state Gaussian HMM on log RV₂₂d)',
                 fontsize=11, fontweight='bold')

    ax = axes[0]
    smove_a = smove.reindex(regime_s.index)
    for r, color in reg_colors.items():
        ax.fill_between(regime_s.index, 0,
                        smove_a.where(regime_s == r).ffill().fillna(0),
                        alpha=0.35, color=color, label=regime_labels.get(r, str(r)))
    smove_a.plot(ax=ax, lw=0.7, color='black', alpha=0.6)
    ax.set_ylabel('surface_move'); ax.legend(fontsize=9, ncol=3)
    ax.set_title('Surface Move with Regime Overlay')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax = axes[1]
    for r, color in reg_colors.items():
        ax.fill_between(regime_s.index, r - 0.4, r + 0.4,
                        where=(regime_s == r).values, color=color, alpha=0.7)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([regime_labels.get(r, str(r)) for r in [0, 1, 2]], fontsize=8)
    ax.set_title('Regime Timeline  (⚠ Viterbi = retrospective)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig(plot_dir / f'{prefix}_hmm_regimes.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_forecast_global(
    targets_df: pd.DataFrame,
    fc_test: Dict,
    fc_wf_ser: pd.Series,
    split_test: pd.Index,
    h_main: int,
    plot_dir: Path,
    prefix: str = '07',
) -> None:
    """Save plots/{prefix}_forecast_global.png."""
    _tgt  = targets_df[f'future_rv_{h_main}d'].reindex(split_test)
    _har  = np.exp(fc_test.get('HAR|global',   pd.Series(dtype=float)).reindex(split_test))
    _harj = np.exp(fc_test.get('HAR-J|global', pd.Series(dtype=float)).reindex(split_test))
    _wf   = np.exp(fc_wf_ser.reindex(split_test))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f'Global Surface RV Forecast  (test set, h={h_main}d)',
                 fontsize=12, fontweight='bold')
    ax = axes[0]
    _tgt.plot(ax=ax, lw=0.7, color='black', alpha=0.7, label=f'actual RV{h_main}')
    _harj.plot(ax=ax, lw=1.0, color='tomato', label='HAR-J')
    _har.plot(ax=ax,  lw=0.9, color='steelblue', ls='--', label='HAR')
    _wf.plot(ax=ax,   lw=0.8, color='darkorange', ls=':', label='HAR-J (WF)')
    ax.set_ylabel('RV'); ax.legend(fontsize=9, ncol=4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.set_title('Forecast vs Actual')

    ax = axes[1]
    (_tgt - _harj).dropna().plot(ax=ax, lw=0.7, color='tomato', alpha=0.8, label='resid HAR-J')
    (_tgt - _har).dropna().plot( ax=ax, lw=0.7, color='steelblue', alpha=0.6, ls='--',
                                 label='resid HAR')
    ax.axhline(0, lw=0.8, color='black', ls=':')
    ax.set_ylabel('residual'); ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.set_title('Forecast Residuals')
    plt.tight_layout()
    plt.savefig(plot_dir / f'{prefix}_forecast_global.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_cstar_term_structure(
    cstar_by_mat: pd.DataFrame,
    a_hat: float,
    b_hat: float,
    r2: float,
    sqrtT_fit: np.ndarray,
    cstar_fit: np.ndarray,
    plot_dir: Path,
    prefix: str = '07',
) -> None:
    """Save plots/{prefix}_cstar_term_structure.png."""
    T_vals    = cstar_by_mat['T'].values
    cstar_v   = cstar_by_mat['c_star'].values
    t_labels  = cstar_by_mat['maturity'].values
    vm        = np.isfinite(cstar_v)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('c* Term Structure  (val-set ES₉₅)', fontsize=11, fontweight='bold')

    ax = axes[0]
    ax.scatter(T_vals[vm], cstar_v[vm], s=80, color='navy', zorder=5, label='c*(T)')
    for t, cs, lbl in zip(T_vals[vm], cstar_v[vm], t_labels[vm]):
        ax.annotate(lbl, (t, cs), textcoords='offset points', xytext=(5, 4), fontsize=8)
    if len(cstar_fit) > 0 and np.isfinite(r2):
        ax.plot(sqrtT_fit**2, cstar_fit, '--', color='tomato', lw=1.3,
                label=f'OLS fit  R²={r2:.2f}')
    ax.set_xlabel('T (years)'); ax.set_ylabel('c*')
    ax.set_title('c* vs Maturity'); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(np.sqrt(T_vals[vm]), cstar_v[vm], s=80, color='steelblue', zorder=5)
    for t, cs, lbl in zip(T_vals[vm], cstar_v[vm], t_labels[vm]):
        ax.annotate(lbl, (np.sqrt(t), cs), textcoords='offset points', xytext=(5, 4), fontsize=8)
    if len(cstar_fit) > 0 and np.isfinite(r2):
        ax.plot(sqrtT_fit, cstar_fit, '--', color='tomato', lw=1.3,
                label=f'{a_hat:.2f} + {b_hat:.2f}×√T')
    ax.set_xlabel('√T'); ax.set_ylabel('c*')
    ax.set_title('c* vs √T  (linearised)'); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir / f'{prefix}_cstar_term_structure.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_spread_decomposition(
    split_test: pd.Index,
    targets_df: pd.DataFrame,
    spread_final: pd.DataFrame,
    spread_AS: pd.DataFrame,
    bt_df: pd.DataFrame,
    plot_dir: Path,
    h_main: int = 5,
    prefix: str = '07',
) -> None:
    """Save plots/{prefix}_spread_decomposition.png and plots/{prefix}_coverage_by_bucket.png."""
    _act = targets_df[f'future_rv_{h_main}d'].reindex(split_test)
    _sf  = spread_final.mean(axis=1).reindex(split_test)
    _as  = spread_AS.mean(axis=1).reindex(split_test)
    _add = _sf - _as

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Spread Decomposition  (global, test set)', fontsize=11, fontweight='bold')
    ax = axes[0]
    _sf.plot(ax=ax,  lw=1.0, color='navy',       label='spread_final')
    _as.plot(ax=ax,  lw=0.9, color='steelblue', ls='--', label='spread_AS')
    _add.plot(ax=ax, lw=0.8, color='darkorange', label='addon c×RV̂', alpha=0.85)
    _act.plot(ax=ax, lw=0.6, color='black', alpha=0.5, label='actual RV₅')
    ax.set_title('Spread Components vs Actual RV  (vol units)')
    ax.legend(fontsize=8, ncol=4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    ax = axes[1]
    covered = (_act <= _sf).reindex(split_test)
    ax.fill_between(split_test, 0, 1, where=covered.fillna(False),
                    alpha=0.35, color='green', label='covered')
    ax.fill_between(split_test, 0, 1, where=(~covered.fillna(True)),
                    alpha=0.35, color='red', label='exceedance')
    ax.set_title('Coverage Timeline  (global)')
    ax.set_ylim(-0.1, 1.2); ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.tight_layout()
    plt.savefig(plot_dir / f'{prefix}_spread_decomposition.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Coverage bar chart
    x   = np.arange(len(bt_df))
    lbl = bt_df['bucket'].values
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Coverage Evaluation Results by Bucket', fontsize=11, fontweight='bold')
    ax = axes[0]
    if 'coverage_AS' in bt_df.columns:
        ax.bar(x - 0.18, bt_df['coverage_AS'].values, 0.32,
               label='Coverage (AS baseline)', color='lightsteelblue', alpha=0.85)
        ax.bar(x + 0.18, bt_df['coverage'].values, 0.32,
               label='Coverage (spread_final)', color='seagreen', alpha=0.85)
        ax.set_title('Coverage: AS Baseline vs spread_final')
    else:
        ax.bar(x, bt_df['coverage'].values, color='seagreen', alpha=0.8)
        ax.set_title('Coverage')
    ax.axhline(0.95, ls='--', color='red', lw=1.2, label='95% target')
    ax.set_xticks(x); ax.set_xticklabels(lbl, rotation=20, fontsize=8)
    ax.legend(fontsize=9); ax.set_ylim(0, 1.05)
    ax = axes[1]
    ax.bar(x - 0.2, bt_df['mean_AS'].values,    0.35, label='mean_AS',    color='steelblue', alpha=0.8)
    ax.bar(x + 0.2, bt_df['mean_addon'].values,  0.35, label='mean_addon', color='darkorange', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(lbl, rotation=20, fontsize=8)
    ax.set_title('Spread Components'); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_dir / f'{prefix}_coverage_by_bucket.png', dpi=150, bbox_inches='tight')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════
# 10. UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def chronological_split(
    index: pd.Index,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """Return (train, val, test) DatetimeIndex splits. No shuffle."""
    n   = len(index)
    nt  = int(n * train_frac)
    nv  = int(n * val_frac)
    return index[:nt], index[nt: nt + nv], index[nt + nv:]


def ensure_dirs(*dirs: Path) -> None:
    """Create directories (and parents) if they do not exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    """Save DataFrame to CSV."""
    df.to_csv(path, index=index)
