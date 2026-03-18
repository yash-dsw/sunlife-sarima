"""
SARIMAX Workforce Cost Forecasting — Multivariate Edition
Sun Life Insurance India | FY 2026 Forecast

Model selection: full grid search over (p,d,q)(P,D,Q,12).
  - Columns in EXOG_MAP are trained as multivariate SARIMAX with exogenous drivers.
  - All other columns remain univariate ARIMA.
  - Best order chosen by composite score: 0.4*norm(AIC) + 0.6*norm(MAPE).

SHAP Explainability:
  - shap.KernelExplainer used for all multivariate models after final refit.
  - SHAP values (per forecast step per feature) and summary (mean |SHAP| per feature)
    are persisted inside each .pkl alongside the model.

Splits:
  Train      : Jan 2022 – Dec 2024  (36 months)
  Test       : Jan 2025 – Dec 2025  (12 months) → model selection + metrics
  Validation : Jan 2026 – Dec 2026  (12 months) → forecast vs actuals

PKL naming  : <DeptName>_<DriverName>_<budget|actual|driver>.pkl
"""

import re
import pickle
import warnings
import logging
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import shap
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────
DATA_FILE  = r"cost_forecast_sample_data_v2_updated 2.xlsx"
MODEL_DIR  = Path("sarimax_models")
RESULT_DIR = Path("sarimax_results")
CHART_DIR  = Path("sarimax_charts")
OUTPUT_CSV = Path("sarimax_full_forecast.csv")

SEASONAL_PERIOD = 12
FORECAST_STEPS  = 12       # Jan–Dec 2026

# Grid search ranges  (p,d,q)(P,D,Q,12)
P_VALUES  = [0, 1, 2]
D_VALUES  = [0, 1, 2]
Q_VALUES  = [0, 1, 2]
SP_VALUES = [0, 1]
SD_VALUES = [0, 1]
SQ_VALUES = [0, 1]

# Composite score weights
W_AIC  = 0.4
W_MAPE = 0.6

# SHAP background sample size (smaller = faster, larger = more accurate)
SHAP_BACKGROUND_SAMPLES = 20
SHAP_NSAMPLES           = 80   # KernelExplainer nsamples per call

MONTH_MAP = {
    "Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
    "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12,
}

# ── Exogenous driver mapping ───────────────────────────────
# target_column -> list of exogenous columns fed into SARIMAX
# Only columns that are either raw data or modelled BEFORE the target column
# (sorted alphabetically in the main loop) should appear here.
EXOG_MAP: dict[str, list[str]] = {
    # Salary-driven actuals
    "Salary Cost (INR)":              ["Closing HC", "Avg Salary (Wtd) (INR)", "Avg Appraisal Rate"],
    "Bonus/Var Pay (INR)":            ["Closing HC", "Salary Cost (INR)"],
    "Overtime Cost (INR)":            ["Total Exits"],
    "Long Leave Cost (INR)":          ["Long Leave Count"],
    # Headcount-driven indirect actuals
    "HR Cost (INR)":                  ["Closing HC"],
    "Admin Overhead (INR)":           ["Closing HC"],
    "IT License Cost (INR)":          ["Closing HC"],
    "IT Equipment Cost (INR)":        ["Total Hires"],
    "Training & Dev Cost (INR)":      ["Total Hires", "Closing HC"],
    "Learning Platform Cost (INR)":   ["Closing HC"],
    "Emp Engagement Cost (INR)":      ["Closing HC"],
    "Mgmt Overhead (INR)":            ["Closing HC"],
    "DBTS Charges (INR)":             ["Closing HC"],
    "Consultancy Charges (INR)":      ["Closing HC"],
    "Financial Operation Charges (INR)": ["Salary Cost (INR)"],
    # WFO-driven actuals
    "Office Rent & Facilities (INR)": ["WFO Count"],
    "Utilities Cost (INR)":           ["WFO Count"],
    # Budget columns
    "Budget: Salary (INR)":           ["Closing HC"],
    "Budget: Long Leave (INR)":       ["Long Leave Count"],
    "Budget: Training & Dev (INR)":   ["Total Hires", "Closing HC", "Billable HC"],
    "Budget: HR Cost (INR)":          ["Closing HC"],
    "Budget: Admin Overhead (INR)":   ["Closing HC"],
    "Budget: IT License (INR)":       ["Closing HC"],
    "Budget: IT Equipment (INR)":     ["Total Hires", "Closing HC"],
    "Budget: Utilities (INR)":        ["WFO Count"],
    "Budget: Mgmt Overhead (INR)":    ["Closing HC"],
    "Budget: Learning Platform (INR)":["Closing HC"],
    "Budget: Emp Engagement (INR)":   ["Closing HC"],
    "Original Budget (INR)":          ["Closing HC", "Total Hires"],
    "Attrition % (Ann.)":             ["Total Exits", "Opening HC"]
}

# ── Columns on which SARIMAX models are trained ───────────
MODEL_COLS = {
    # Headcount & workforce drivers
    "Opening HC", "Total Hires", "Total Exits", "Long Leave Count",
    "Closing HC", "WFH Count", "WFO Count", "Billable HC", "Talent Pool",
    "Band 1 Count", "Band 2 Count", "Band 3 Count",
    "Band 4 Count", "Band 5 Count", "Band 6 Count",
    "Attrition % (Ann.)", "Avg Appraisal Rate", "CAD/INR Forex Rate",
    # Salary rates per band + weighted average
    "Band 1 Sal/FTE (INR)", "Band 2 Sal/FTE (INR)", "Band 3 Sal/FTE (INR)",
    "Band 4 Sal/FTE (INR)", "Band 5 Sal/FTE (INR)", "Band 6 Sal/FTE (INR)",
    "Avg Salary (Wtd) (INR)",
    # Direct cost – actual & budget pairs
    "Budget: Salary (INR)",       "Salary Cost (INR)",
    "Budget: Bonus (INR)",        "Bonus/Var Pay (INR)",
    "Budget: Overtime (INR)",     "Overtime Cost (INR)",
    "Budget: Long Leave (INR)",   "Long Leave Cost (INR)",
    # Indirect cost – actual & budget pairs
    "Budget: Training & Dev (INR)",    "Training & Dev Cost (INR)",
    "Budget: HR Cost (INR)",           "HR Cost (INR)",
    "Budget: Admin Overhead (INR)",    "Admin Overhead (INR)",
    "Budget: IT License (INR)",        "IT License Cost (INR)",
    "Budget: IT Equipment (INR)",      "IT Equipment Cost (INR)",
    "Budget: Office Rent (INR)",       "Office Rent & Facilities (INR)",
    "Budget: Utilities (INR)",         "Utilities Cost (INR)",
    "Budget: Mgmt Overhead (INR)",     "Mgmt Overhead (INR)",
    "Budget: Learning Platform (INR)", "Learning Platform Cost (INR)",
    "Budget: Emp Engagement (INR)",    "Emp Engagement Cost (INR)",
    "Budget: DBTS Charges (INR)",      "DBTS Charges (INR)",
    "Budget: Consultancy (INR)",       "Consultancy Charges (INR)",
    "Budget: Financial Operation Charges (INR)",   "Financial Operation Charges (INR)",
    # Budget total
    "Original Budget (INR)",
}

# ── Direct cost components (business logic sum) ───────────
DIRECT_ACTUAL_COLS = [
    "Salary Cost (INR)", "Benefits Cost (INR)", "Bonus/Var Pay (INR)",
    "Payroll Tax (INR)", "Travel Allowance (INR)", "Meal Allowance (INR)",
    "Overtime Cost (INR)", "Recruitment Cost (INR)", "Long Leave Cost (INR)",
]

# ── Indirect cost components ───────────────────────────────
INDIRECT_ACTUAL_COLS = [
    "Training & Dev Cost (INR)", "HR Cost (INR)", "Admin Overhead (INR)",
    "IT License Cost (INR)", "IT Equipment Cost (INR)",
    "Office Rent & Facilities (INR)", "Utilities Cost (INR)",
    "Mgmt Overhead (INR)", "Learning Platform Cost (INR)",
    "Emp Engagement Cost (INR)", "DBTS Charges (INR)",
    "Consultancy Charges (INR)", "Financial Operation Charges (INR)",
]

# ── Ratio-derived cols (business logic, no model) ─────────
RATIO_DERIVED = {
    "Benefits Cost (INR)":    ("Salary Cost (INR)",   "Budget: Benefits (INR)"),
    "Payroll Tax (INR)":      ("Salary Cost (INR)",   "Budget: Payroll Tax (INR)"),
    "Travel Allowance (INR)": ("Closing HC",          "Budget: Travel Allow (INR)"),
    "Meal Allowance (INR)":   ("Closing HC",          "Budget: Meal Allow (INR)"),
    "Recruitment Cost (INR)": ("Total Hires",         "Budget: Recruitment (INR)"),
    "Financial Operation Charges (INR)": (
        "Salary Cost (INR)", "Budget: Financial Operation Charges (INR)"),
}


# ─────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────

def safe_name(s: str) -> str:
    return re.sub(r"[^\w]", "_", str(s)).strip("_")


def pkl_name(dept: str, driver: str, kind: str) -> str:
    return f"{safe_name(dept)}_{safe_name(driver)}_{kind}.pkl"


def load_sheet(filepath: str, sheet: str) -> pd.DataFrame:
    raw     = pd.read_excel(filepath, sheet_name=sheet, header=None)
    headers = raw.iloc[1].tolist()
    data    = raw.iloc[2:].copy()
    data.columns = headers
    data = data[data["Month"].isin(MONTH_MAP)].copy()
    data["_date"] = pd.to_datetime(
        data["Year"].astype(int).astype(str) + "-" +
        data["Month"].map(MONTH_MAP).astype(str) + "-01"
    )
    data = data.sort_values("_date").set_index("_date")
    data.index = pd.DatetimeIndex(data.index, freq="MS")
    for c in data.columns:
        if c not in ("Month", "Year"):
            data[c] = pd.to_numeric(data[c], errors="coerce")
    return data


def hint_d(series: pd.Series) -> int:
    """ADF-based hint for d (0/1/2)."""
    s = series.dropna()
    if len(s) < 10:
        return 1
    try:
        if adfuller(s, autolag="AIC")[1] < 0.05:
            return 0
        if adfuller(s.diff().dropna(), autolag="AIC")[1] < 0.05:
            return 1
    except Exception:
        pass
    return 2


def build_exog(df: pd.DataFrame, exog_cols: list[str], idx: pd.Index) -> np.ndarray | None:
    """
    Extract exogenous columns from df aligned to idx.
    Forward-fills then back-fills gaps. Returns float 2D array or None.
    """
    if not exog_cols:
        return None
    present = [c for c in exog_cols if c in df.columns]
    if not present:
        return None
    sub = df.loc[idx, present].astype(float).ffill().bfill()
    if sub.empty or sub.isnull().all().all():
        return None
    return sub.values


def build_exog_from_fc(fc_dict: dict, exog_cols: list[str],
                        df: pd.DataFrame, fc_index: pd.DatetimeIndex,
                        n: int) -> np.ndarray | None:
    """
    Build exog matrix for the forecast horizon.
    Uses forecasted values from fc_dict when available,
    falling back to trailing average from df for any missing column.
    """
    if not exog_cols:
        return None
    rows = []
    for col in exog_cols:
        if col in fc_dict:
            arr = np.nan_to_num(fc_dict[col][:n].astype(float))
        elif col in df.columns:
            arr = np.full(n, df[col].dropna().tail(12).mean())
        else:
            arr = np.zeros(n)
        rows.append(arr)
    if not rows:
        return None
    return np.column_stack(rows)  # shape (n, n_features)


def calc_mape(actual: np.ndarray, pred: np.ndarray) -> float:
    a, p = np.array(actual, dtype=float), np.array(pred, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(p) & (a != 0)
    if mask.sum() == 0:
        return np.inf
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


def calc_metrics(actual: np.ndarray, pred: np.ndarray):
    a, p = np.array(actual, dtype=float), np.array(pred, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(p)
    a, p = a[mask], p[mask]
    if len(a) == 0:
        return np.nan, np.nan, np.nan
    mae  = np.mean(np.abs(a - p))
    rmse = np.sqrt(np.mean((a - p) ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.nanmean(np.abs((a - p) / np.where(a == 0, np.nan, a))) * 100
    return round(float(mae), 4), round(float(rmse), 4), round(float(mape), 4)


# ─────────────────────────────────────────────────────────
# DUAL-OBJECTIVE GRID SEARCH  (now exog-aware)
# ─────────────────────────────────────────────────────────

def grid_search_sarimax(train: pd.Series, test: pd.Series, hint_d_val: int,
                         exog_train: np.ndarray | None = None,
                         exog_test: np.ndarray | None = None):
    """
    Exhaustive grid search over (p,d,q)(P,D,Q,12).
    Passes exog_train / exog_test into SARIMAX when provided.

    Each candidate is scored by:
        composite = W_AIC * norm_aic + W_MAPE * norm_mape

    Returns: (best_fitted_model, best_order, best_seasonal_order,
               best_aic, best_mape, candidates_df)
    """
    candidates = []

    d_order = sorted(D_VALUES, key=lambda x: abs(x - hint_d_val))

    for p, d, q, sp, sd, sq in product(P_VALUES, d_order, Q_VALUES,
                                        SP_VALUES, SD_VALUES, SQ_VALUES):
        if d + sd > 2:
            continue
        try:
            mdl = SARIMAX(
                train,
                exog=exog_train,
                order=(p, d, q),
                seasonal_order=(sp, sd, sq, SEASONAL_PERIOD),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=300, method="lbfgs")

            aic = mdl.aic
            fc  = mdl.get_forecast(steps=len(test), exog=exog_test).predicted_mean.values
            mpe = calc_mape(test.values, fc)

            if np.isfinite(aic) and np.isfinite(mpe):
                candidates.append({
                    "order":          (p, d, q),
                    "seasonal_order": (sp, sd, sq, SEASONAL_PERIOD),
                    "aic":  aic,
                    "mape": mpe,
                    "model": mdl,
                })
        except Exception:
            continue

    if not candidates:
        d_fb = hint_d_val
        try:
            mdl_fb = SARIMAX(
                train, exog=exog_train, order=(1, d_fb, 1),
                seasonal_order=(0, 0, 0, SEASONAL_PERIOD),
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False, maxiter=300)
        except Exception:
            mdl_fb = SARIMAX(
                train, order=(1, d_fb, 1),
                seasonal_order=(0, 0, 0, SEASONAL_PERIOD),
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False, maxiter=300)
            exog_test = None
        fc_fb  = mdl_fb.get_forecast(len(test), exog=exog_test).predicted_mean.values
        mpe_fb = calc_mape(test.values, fc_fb)
        return (mdl_fb, (1, d_fb, 1), (0, 0, 0, SEASONAL_PERIOD),
                mdl_fb.aic, mpe_fb, pd.DataFrame())

    df_cand = pd.DataFrame([{k: v for k, v in c.items() if k != "model"}
                             for c in candidates])

    def minmax(col):
        mn, mx = col.min(), col.max()
        return (col - mn) / (mx - mn) if mx > mn else pd.Series(np.zeros(len(col)))

    df_cand["norm_aic"]  = minmax(df_cand["aic"])
    df_cand["norm_mape"] = minmax(df_cand["mape"])
    df_cand["score"]     = W_AIC * df_cand["norm_aic"] + W_MAPE * df_cand["norm_mape"]

    best_idx = int(df_cand["score"].idxmin())
    best_c   = candidates[best_idx]

    return (best_c["model"], best_c["order"], best_c["seasonal_order"],
            best_c["aic"], best_c["mape"], df_cand)


# ─────────────────────────────────────────────────────────
# SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────

def compute_shap(fitted_model, exog_background: np.ndarray,
                  exog_forecast: np.ndarray,
                  feature_names: list[str]) -> tuple:
    """
    Compute SHAP values for multivariate SARIMAX using KernelExplainer.

    Parameters
    ----------
    fitted_model     : SARIMAXResultsWrapper — final fitted model
    exog_background  : (n_bg, n_features) array — historical exog for background
    exog_forecast    : (forecast_steps, n_features) array — exog for 2026 forecast
    feature_names    : list of column names corresponding to exog features

    Returns
    -------
    shap_values  : ndarray (forecast_steps, n_features)
    shap_summary : dict {feature_name: mean_abs_shap}
    """
    n_bg = min(SHAP_BACKGROUND_SAMPLES, len(exog_background))
    # Subsample background for speed
    bg_idx   = np.linspace(0, len(exog_background) - 1, n_bg, dtype=int)
    bg_data  = exog_background[bg_idx]

    def predict_fn(X: np.ndarray) -> np.ndarray:
        """Given a 2D matrix of exog rows, return 1-step-ahead forecasts."""
        preds = []
        for row in X:
            try:
                # apply() + append + forecast is the statsmodels-recommended
                # way to get a conditional one-step forecast with new exog.
                fc_val = float(
                    fitted_model.get_forecast(steps=1, exog=row.reshape(1, -1))
                    .predicted_mean.iloc[0]
                )
            except Exception:
                fc_val = float(fitted_model.fittedvalues.iloc[-1])
            preds.append(fc_val)
        return np.array(preds, dtype=float)

    try:
        explainer  = shap.KernelExplainer(predict_fn, bg_data)
        shap_vals  = explainer.shap_values(exog_forecast,
                                            nsamples=SHAP_NSAMPLES, silent=True)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        shap_vals = np.array(shap_vals, dtype=float)

        summary = {
            feat: round(float(np.mean(np.abs(shap_vals[:, i]))), 6)
            for i, feat in enumerate(feature_names)
        }
        # Sort descending by importance
        summary = dict(sorted(summary.items(), key=lambda x: -x[1]))
        return shap_vals, summary
    except Exception as e:
        log.warning("    SHAP computation failed: %s", e)
        return None, {}


def plot_shap_bar(shap_summary: dict, sheet: str, col: str, chart_dir: Path):
    """Save a horizontal bar chart of mean |SHAP| per feature."""
    if not shap_summary:
        return
    feats  = list(shap_summary.keys())
    values = [shap_summary[f] for f in feats]

    fig, ax = plt.subplots(figsize=(8, max(3, len(feats) * 0.55)))
    bars = ax.barh(feats[::-1], values[::-1], color="#2563EB", alpha=0.85)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("Mean |SHAP value| (avg impact on forecast)", fontsize=9)
    ax.set_title(f"SHAP Feature Importance\n{sheet} | {col}", fontsize=10,
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fname = chart_dir / f"{safe_name(sheet)}_{safe_name(col)}_shap.png"
    fig.savefig(fname, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────
# BUSINESS LOGIC HELPERS
# ─────────────────────────────────────────────────────────

def derive_by_ratio(fc_dict, driver, numerator, hist_df, n):
    if numerator not in fc_dict or driver not in hist_df.columns:
        return np.full(n, np.nan)
    hist_num = hist_df[numerator].replace(0, np.nan)
    hist_drv = hist_df[driver]
    common   = hist_num.dropna().index.intersection(hist_drv.dropna().index)
    if len(common) == 0:
        return np.full(n, np.nan)
    ratios    = (hist_drv.loc[common] / hist_num.loc[common]) \
                    .replace([np.inf, -np.inf], np.nan).dropna()
    avg_ratio = ratios[-36:].mean() if len(ratios) >= 12 else ratios.mean()
    return np.array(fc_dict[numerator]) * avg_ratio


def sum_cols(fc_dict, cols, hist_df, n):
    out = np.zeros(n)
    for c in cols:
        if c in fc_dict:
            out += np.nan_to_num(fc_dict[c])
        elif c in hist_df.columns:
            avg = hist_df[c].dropna().tail(12).mean()
            out += avg if np.isfinite(avg) else 0.0
    return out


# ─────────────────────────────────────────────────────────
# CHART — time-series
# ─────────────────────────────────────────────────────────

def plot_series(sheet, col, hist_series, fc_months, fc_vals, chart_dir):
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(hist_series.index, hist_series.values,
            color="#2563EB", linewidth=1.6, label="Actuals (2022–2025)")
    fc_s = pd.Series(np.array(fc_vals, dtype=float),
                     index=pd.DatetimeIndex(fc_months))
    ax.plot(fc_s.index, fc_s.values,
            color="#DC2626", linewidth=1.8, linestyle="--",
            marker="o", markersize=4, label="SARIMAX Forecast (2026)")
    ax.axvline(pd.Timestamp("2026-01-01"), color="grey",
               linestyle=":", linewidth=1, alpha=0.7)
    ax.set_title(f"{sheet}  |  {col}", fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = chart_dir / f"{safe_name(sheet)}_{safe_name(col)}.png"
    fig.savefig(fname, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    for d in [MODEL_DIR, RESULT_DIR, CHART_DIR]:
        d.mkdir(exist_ok=True)

    xl          = pd.ExcelFile(DATA_FILE)
    all_rows    = []
    metric_rows = []

    for sheet in xl.sheet_names:
        log.info("=" * 70)
        log.info("SHEET: %s", sheet)
        log.info("=" * 70)

        dept_safe = safe_name(sheet)
        (MODEL_DIR / dept_safe).mkdir(exist_ok=True)
        (CHART_DIR / dept_safe).mkdir(exist_ok=True)

        df = load_sheet(DATA_FILE, sheet)

        train_df      = df.loc["2022-01":"2024-12"]
        test_df       = df.loc["2025-01":"2025-12"]
        val_df        = df.loc["2026-01":"2026-12"]
        full_train_df = df.loc["2022-01":"2025-12"]

        fc_months = pd.date_range("2026-01-01", periods=FORECAST_STEPS, freq="MS")
        fc_dict: dict[str, np.ndarray] = {}

        # ── 1. SARIMAX model per column ────────────────────────
        for col in sorted(MODEL_COLS):
            if col not in df.columns:
                continue

            train = train_df[col].dropna()
            test  = test_df[col].dropna()

            if len(train) < 18 or len(test) < 6:
                log.warning("  [SKIP – insufficient data] %s", col)
                continue

            # ── Resolve exog ──
            exog_cols    = EXOG_MAP.get(col, [])
            exog_cols    = [c for c in exog_cols if c in df.columns]
            exog_train   = build_exog(train_df, exog_cols, train.index)
            exog_test    = build_exog(test_df,  exog_cols, test.index)
            is_multiv    = exog_train is not None

            mode_tag = f"SARIMAX[{'+'.join(exog_cols)}]" if is_multiv else "SARIMA(univariate)"
            log.info("  ▶ %s: %s", mode_tag, col)

            d_hint = hint_d(train)

            # ── Phase 1: grid search on train→test ──
            best_mdl, best_ord, best_s_ord, best_aic, best_mape, _ = \
                grid_search_sarimax(train, test, d_hint, exog_train, exog_test)

            test_pred           = best_mdl.get_forecast(len(test), exog=exog_test).predicted_mean.values
            mae, rmse, mape_out = calc_metrics(test.values, test_pred)

            log.info("    Best order: %s  Seasonal: %s | AIC: %.2f  MAPE: %.2f%%  MAE: %.4f",
                     best_ord, best_s_ord, best_aic, mape_out if np.isfinite(mape_out) else 0.0, mae)

            # ── Phase 2: refit on full 2022-2025 data ──
            full_train   = full_train_df[col].dropna()
            exog_full    = build_exog(full_train_df, exog_cols, full_train.index)

            try:
                final_mdl = SARIMAX(
                    full_train,
                    exog=exog_full,
                    order=best_ord,
                    seasonal_order=best_s_ord,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False, maxiter=300, method="lbfgs")
                final_aic = final_mdl.aic
            except Exception as e:
                log.warning("    Final refit failed (%s) – keeping phase-1 model", e)
                final_mdl = best_mdl
                final_aic = best_aic
                exog_full  = exog_train   # fall back

            # ── Build exog for 2026 forecast from previously forecasted values ──
            exog_fc = build_exog_from_fc(fc_dict, exog_cols, full_train_df,
                                          fc_months, FORECAST_STEPS)

            fc_pred = final_mdl.get_forecast(FORECAST_STEPS, exog=exog_fc).predicted_mean.values
            fc_dict[col] = fc_pred

            # ── SHAP (only for multivariate models) ──
            shap_vals, shap_summary = None, {}
            if is_multiv and exog_full is not None and exog_fc is not None:
                log.info("    Computing SHAP values for %d exog features …", len(exog_cols))
                shap_vals, shap_summary = compute_shap(
                    final_mdl,
                    exog_background=exog_full,
                    exog_forecast=exog_fc,
                    feature_names=exog_cols,
                )
                if shap_summary:
                    log.info("    SHAP summary: %s", shap_summary)
                    plot_shap_bar(shap_summary, sheet, col, CHART_DIR / dept_safe)

            # ── PKL kind tag ──
            if "Budget:" in col:
                kind = "budget"
            elif any(k in col for k in [
                "Sal/FTE", "Count", "HC", "Hires", "Exits", "Leave",
                "Attrition", "Appraisal", "Forex", "WFH", "WFO",
                "Billable", "Talent", "Avg Salary",
            ]):
                kind = "driver"
            else:
                kind = "actual"

            # ── Save model + SHAP ──
            pkl_path = MODEL_DIR / dept_safe / pkl_name(dept_safe, col, kind)
            with open(pkl_path, "wb") as f:
                pickle.dump({
                    "model":           final_mdl,
                    "order":           best_ord,
                    "seasonal_order":  best_s_ord,
                    "aic":             round(final_aic, 4),
                    "selection_mape":  round(best_mape, 4),
                    "sheet":           sheet,
                    "column":          col,
                    "kind":            kind,
                    "train_end":       "2025-12",
                    "w_aic":           W_AIC,
                    "w_mape":          W_MAPE,
                    # Multivariate / SHAP fields
                    "is_multivariate": is_multiv,
                    "feature_names":   exog_cols,
                    "shap_values":     shap_vals,        # ndarray or None
                    "shap_summary":    shap_summary,     # {feat: mean_abs_shap}
                }, f)

            metric_rows.append({
                "Sheet":           sheet,
                "Column":          col,
                "Kind":            kind,
                "Multivariate":    is_multiv,
                "Exog_Features":   "|".join(exog_cols) if exog_cols else "",
                "p":               best_ord[0],
                "d":               best_ord[1],
                "q":               best_ord[2],
                "P":               best_s_ord[0],
                "D":               best_s_ord[1],
                "Q":               best_s_ord[2],
                "m":               best_s_ord[3],
                "AIC":             round(final_aic, 2),
                "Selection_MAPE":  round(best_mape, 2),
                "Test_MAE":        mae,
                "Test_RMSE":       rmse,
                "Test_MAPE_%":     mape_out,
                "Top_SHAP_Feature": next(iter(shap_summary), "") if shap_summary else "",
                "Top_SHAP_Value":   next(iter(shap_summary.values()), None) if shap_summary else None,
            })

            # Chart — time series
            plot_series(sheet, col, full_train_df[col].dropna(),
                        fc_months, fc_pred, CHART_DIR / dept_safe)

        # ── 2. Ratio-derived columns ─────────────────────────
        log.info("  ▶ Deriving ratio-based columns ...")
        for drv_col, (num_col, _) in RATIO_DERIVED.items():
            if drv_col in fc_dict:
                continue
            derived      = derive_by_ratio(fc_dict, drv_col, num_col,
                                           full_train_df, FORECAST_STEPS)
            fc_dict[drv_col] = derived
            if drv_col in full_train_df.columns:
                plot_series(sheet, drv_col, full_train_df[drv_col].dropna(),
                            fc_months, derived, CHART_DIR / dept_safe)

        # ── 3. Business logic aggregates ─────────────────────
        log.info("  ▶ Calculating cost aggregates ...")

        total_direct   = sum_cols(fc_dict, DIRECT_ACTUAL_COLS,   full_train_df, FORECAST_STEPS)
        total_indirect = sum_cols(fc_dict, INDIRECT_ACTUAL_COLS, full_train_df, FORECAST_STEPS)
        total_actual   = total_direct + total_indirect

        fc_dict["Total Direct Cost (INR)"]   = total_direct
        fc_dict["Total Indirect Cost (INR)"] = total_indirect
        fc_dict["Total Actual Cost (INR)"]   = total_actual

        if "Original Budget (INR)" in fc_dict:
            fc_dict["Monthly Variance (INR)"] = (
                np.nan_to_num(fc_dict["Original Budget (INR)"]) - total_actual
            )

        for tc in ["Total Direct Cost (INR)", "Total Indirect Cost (INR)",
                   "Total Actual Cost (INR)", "Monthly Variance (INR)"]:
            if tc in full_train_df.columns:
                plot_series(sheet, tc, full_train_df[tc].dropna(),
                            fc_months, fc_dict.get(tc, np.full(12, np.nan)),
                            CHART_DIR / dept_safe)

        # ── 4. Collect output rows ────────────────────────────
        for i, dt in enumerate(fc_months):
            row = {
                "Sheet": sheet,
                "Date":  dt.strftime("%Y-%m"),
                "Month": dt.strftime("%b"),
                "Year":  2026,
            }
            for col, fc_arr in fc_dict.items():
                fc_val  = float(fc_arr[i]) if np.isfinite(fc_arr[i]) else np.nan
                act_val = np.nan
                if dt in val_df.index and col in val_df.columns:
                    v = val_df.loc[dt, col]
                    act_val = float(v) if pd.notna(v) else np.nan
                row[f"Forecast_{col}"] = round(fc_val,  4) if pd.notna(fc_val)  else np.nan
                row[f"Actual_{col}"]   = round(act_val, 4) if pd.notna(act_val) else np.nan
            all_rows.append(row)

    # ── 5. Save outputs ───────────────────────────────────────
    pd.DataFrame(all_rows).to_csv(OUTPUT_CSV, index=False)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(RESULT_DIR / "model_evaluation_metrics.csv", index=False)

    # ── 6. Print evaluation table ─────────────────────────────
    pd.set_option("display.max_colwidth", 42)
    pd.set_option("display.float_format",  "{:.2f}".format)
    pd.set_option("display.width", 180)

    print("\n" + "═" * 180)
    print("  SARIMAX MODEL EVALUATION  |  Multivariate Edition  |"
          "  Selection: min(0.4·AIC + 0.6·MAPE)  |  Hold-out: Jan–Dec 2025")
    print("═" * 180)

    show_cols = ["Sheet", "Column", "Multivariate", "Exog_Features", "Kind",
                 "p", "d", "q", "P", "D", "Q",
                 "AIC", "Selection_MAPE", "Test_MAE", "Test_RMSE", "Test_MAPE_%",
                 "Top_SHAP_Feature", "Top_SHAP_Value"]
    print(
        metrics_df[show_cols]
        .sort_values(["Sheet", "Test_MAPE_%"])
        .to_string(index=False)
    )
    print("═" * 180)

    print("\n── Per-Sheet Summary ──────────────────────────────────────")
    summary = (
        metrics_df.groupby("Sheet")[["AIC", "Test_MAPE_%", "Test_MAE", "Test_RMSE"]]
        .agg(["mean", "median", "min", "max"])
        .round(2)
    )
    print(summary.to_string())

    mv_count  = metrics_df["Multivariate"].sum()
    univ_count = (~metrics_df["Multivariate"]).sum()
    print(f"\n  Multivariate models : {mv_count}")
    print(f"  Univariate models   : {univ_count}")

    print(f"\n✓  Models       → {MODEL_DIR}/")
    print(f"✓  Charts       → {CHART_DIR}/")
    print(f"✓  Metrics CSV  → {RESULT_DIR}/model_evaluation_metrics.csv")
    print(f"✓  Forecast CSV → {OUTPUT_CSV}\n")


if __name__ == "__main__":
    main()