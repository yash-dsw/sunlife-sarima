"""
SARIMAX Workforce Cost Forecasting
Sun Life Insurance India | FY 2026 Forecast

Model selection: full grid search over (p,d,q)(P,D,Q,12).
  For every candidate order the model is evaluated on the 2025 hold-out set.
  The best order is chosen by a composite score:
      score = w_aic * norm(AIC) + w_mape * norm(MAPE)
  where norm() min-max normalises across all valid candidates for that series
  and weights are w_aic=0.4, w_mape=0.6  (MAPE weighted higher – predictive quality first).

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
DATA_FILE  = r"cost_forecast_sample_data_v2_updated 2.xlsx"   # update path if needed
MODEL_DIR  = Path("sarimax_models")
RESULT_DIR = Path("sarimax_results")
CHART_DIR  = Path("sarimax_charts")
OUTPUT_CSV = Path("sarimax_full_forecast.csv")

SEASONAL_PERIOD = 12
FORECAST_STEPS  = 12       # Jan–Dec 2026

# Grid search ranges  (p,d,q)(P,D,Q,12)
P_VALUES  = [0, 1, 2]
D_VALUES  = [0, 1, 2]      # d is also grid-searched (ADF used only as a hint)
Q_VALUES  = [0, 1, 2]
SP_VALUES = [0, 1]
SD_VALUES = [0, 1]
SQ_VALUES = [0, 1]

# Composite score weights  (must sum to 1)
W_AIC  = 0.4
W_MAPE = 0.6

MONTH_MAP = {
    "Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
    "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12,
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
    "Budget: Finance Charges (INR)",   "Finance Charges (INR)",
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
# key -> (numerator_col, budget_col_for_reference)
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
    """ADF-based hint for d (0/1/2). Grid search still tries all D_VALUES."""
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
        mape = np.mean(np.abs((a - p) / np.where(a == 0, np.nan, a))) * 100
    return round(float(mae), 4), round(float(rmse), 4), round(float(mape), 4)


# ─────────────────────────────────────────────────────────
# DUAL-OBJECTIVE GRID SEARCH
# ─────────────────────────────────────────────────────────

def grid_search_sarimax(train: pd.Series, test: pd.Series, hint_d_val: int):
    """
    Exhaustive grid search over (p,d,q)(P,D,Q,12).

    Each candidate is scored by:
        composite = W_AIC * norm_aic + W_MAPE * norm_mape
    where norm() is min-max normalisation across all valid candidates.

    Returns: (best_fitted_model, best_order_tuple, best_aic, best_mape, all_results_df)
    """
    candidates = []   # list of dicts: order, seasonal_order, aic, mape, model

    # Prioritise the ADF-hinted d to reduce search time but still try all
    d_order = sorted(D_VALUES, key=lambda x: abs(x - hint_d_val))

    total_combos = len(P_VALUES) * len(d_order) * len(Q_VALUES) * \
                   len(SP_VALUES) * len(SD_VALUES) * len(SQ_VALUES)
    log.debug("      Grid: %d combinations", total_combos)

    for p, d, q, sp, sd, sq in product(P_VALUES, d_order, Q_VALUES,
                                        SP_VALUES, SD_VALUES, SQ_VALUES):
        # Skip over-differenced combos
        if d + sd > 2:
            continue
        try:
            mdl = SARIMAX(
                train,
                order=(p, d, q),
                seasonal_order=(sp, sd, sq, SEASONAL_PERIOD),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=300, method="lbfgs")

            aic = mdl.aic
            # Evaluate on test set
            fc  = mdl.get_forecast(steps=len(test)).predicted_mean.values
            mpe = calc_mape(test.values, fc)

            if np.isfinite(aic) and np.isfinite(mpe):
                candidates.append({
                    "order": (p, d, q),
                    "seasonal_order": (sp, sd, sq, SEASONAL_PERIOD),
                    "aic": aic,
                    "mape": mpe,
                    "model": mdl,
                })
        except Exception:
            continue

    if not candidates:
        # Hard fallback
        d_fb = hint_d_val
        mdl_fb = SARIMAX(
            train, order=(1, d_fb, 1), seasonal_order=(0, 0, 0, SEASONAL_PERIOD),
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False, maxiter=300)
        fc_fb  = mdl_fb.get_forecast(len(test)).predicted_mean.values
        mpe_fb = calc_mape(test.values, fc_fb)
        return (mdl_fb, (1, d_fb, 1), (0, 0, 0, SEASONAL_PERIOD),
                mdl_fb.aic, mpe_fb, pd.DataFrame())

    df_cand = pd.DataFrame([{k: v for k, v in c.items() if k != "model"}
                             for c in candidates])

    # Min-max normalise AIC and MAPE
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
# CHART
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
        full_train_df = df.loc["2022-01":"2025-12"]   # used for final model

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

            log.info("  ▶ Grid search SARIMAX: %s", col)
            d_hint = hint_d(train)

            # ── Phase 1: grid search on train→test to find best (p,d,q)(P,D,Q) ─
            best_mdl, best_ord, best_s_ord, best_aic, best_mape, cand_df = \
                grid_search_sarimax(train, test, d_hint)

            # Test metrics with best order
            test_pred           = best_mdl.get_forecast(len(test)).predicted_mean.values
            mae, rmse, mape_out = calc_metrics(test.values, test_pred)

            log.info("    Best order   : %s  Seasonal: %s", best_ord, best_s_ord)
            log.info("    Score → AIC  : %.2f   MAPE: %.2f%%   MAE: %.4f   RMSE: %.4f",
                     best_aic, mape_out, mae, rmse)

            # ── Phase 2: refit best order on full 2022-2025 data ──
            full_train = full_train_df[col].dropna()
            try:
                final_mdl = SARIMAX(
                    full_train,
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

            fc_pred = final_mdl.get_forecast(FORECAST_STEPS).predicted_mean.values
            fc_dict[col] = fc_pred

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

            # ── Save model ──
            pkl_path = MODEL_DIR / dept_safe / pkl_name(dept_safe, col, kind)
            with open(pkl_path, "wb") as f:
                pickle.dump({
                    "model":          final_mdl,
                    "order":          best_ord,
                    "seasonal_order": best_s_ord,
                    "aic":            round(final_aic, 4),
                    "selection_mape": round(best_mape, 4),
                    "sheet":          sheet,
                    "column":         col,
                    "kind":           kind,
                    "train_end":      "2025-12",
                    "w_aic":          W_AIC,
                    "w_mape":         W_MAPE,
                }, f)

            metric_rows.append({
                "Sheet":          sheet,
                "Column":         col,
                "Kind":           kind,
                "p":              best_ord[0],
                "d":              best_ord[1],
                "q":              best_ord[2],
                "P":              best_s_ord[0],
                "D":              best_s_ord[1],
                "Q":              best_s_ord[2],
                "m":              best_s_ord[3],
                "AIC":            round(final_aic, 2),
                "Selection_MAPE": round(best_mape, 2),
                "Test_MAE":       mae,
                "Test_RMSE":      rmse,
                "Test_MAPE_%":    mape_out,
            })

            # Chart
            plot_series(sheet, col, full_train_df[col].dropna(),
                        fc_months, fc_pred, CHART_DIR / dept_safe)

        # ── 2. Ratio-derived columns ────────────────────────────
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

        # ── 3. Business logic aggregates ────────────────────────
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

        # ── 4. Collect output rows ───────────────────────────────
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

    # ── 5. Save outputs ──────────────────────────────────────
    pd.DataFrame(all_rows).to_csv(OUTPUT_CSV, index=False)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(RESULT_DIR / "model_evaluation_metrics.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(RESULT_DIR / "model_summary.csv", index=False)

    # ── 6. Print evaluation table ────────────────────────────
    pd.set_option("display.max_colwidth", 42)
    pd.set_option("display.float_format",  "{:.2f}".format)
    pd.set_option("display.width", 140)

    print("\n" + "═" * 140)
    print("  SARIMAX MODEL EVALUATION  |  Selection: min(0.4·AIC + 0.6·MAPE)"
          "  |  Hold-out: Jan–Dec 2025")
    print("═" * 140)

    show_cols = ["Sheet", "Column", "Kind",
                 "p", "d", "q", "P", "D", "Q",
                 "AIC", "Selection_MAPE",
                 "Test_MAE", "Test_RMSE", "Test_MAPE_%"]
    print(
        metrics_df[show_cols]
        .sort_values(["Sheet", "Test_MAPE_%"])
        .to_string(index=False)
    )
    print("═" * 140)

    print("\n── Per-Sheet Summary ───────────────────────────────────────")
    summary = (
        metrics_df.groupby("Sheet")[["AIC", "Test_MAPE_%", "Test_MAE", "Test_RMSE"]]
        .agg(["mean", "median", "min", "max"])
        .round(2)
    )
    print(summary.to_string())

    print(f"\n✓  Models      → {MODEL_DIR}/")
    print(f"✓  Charts      → {CHART_DIR}/")
    print(f"✓  Metrics CSV → {RESULT_DIR}/model_evaluation_metrics.csv")
    print(f"✓  Forecast CSV→ {OUTPUT_CSV}\n")


if __name__ == "__main__":
    main()