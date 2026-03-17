"""
evaluate.py
===========
Validation & testing script for the Workforce Cost Forecasting model.

Tests two things:
  1. SARIMA series accuracy  — HC, Attrition%, Avg Salary, Engagement, Other Indirect
  2. End-to-end cost accuracy — predicted total cost vs actual total cost from Excel
     (both individual lines AND totals)

Walk-forward rounds
-------------------
  test25: Train Jan-22 → Dec-24  |  Test Jan-25 → Dec-25
  val26:  Train Jan-22 → Dec-25  |  Validate Jan-26 → Dec-26

Metrics reported per series per dept
-------------------------------------
  MAPE     — Mean Absolute Percentage Error
  MAE      — Mean Absolute Error (INR)
  RMSE     — Root Mean Squared Error (INR)
  DirAcc   — Direction Accuracy % (got up/down trend right?)
  MaxErr   — Worst single-month absolute error

Usage
-----
    python evaluate.py
    python evaluate.py --dept Technology
    python evaluate.py --dept Technology --rounds test25 val26 --output report.csv
"""

from __future__ import annotations
import argparse
import warnings
import sys
from pathlib import Path
from typing import Any

# Force UTF-8 output so ₹ / unicode symbols work on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"


def _default_data_path(base_dir: Path) -> Path:
    samples = sorted(base_dir.glob("cost_forecast_sample_data*.xlsx"))
    if samples:
        return samples[0]
    return base_dir / "data" / "raw" / "workforce_cost_model_v4.xlsx"

sys.path.insert(0, str(BASE_DIR))
from src.data_loader    import load_all, DEPARTMENTS
from src.model_trainer  import fit_best_sarima, learn_series_strategy, _resolve_series_map
from src.cost_rules     import learn_rates, compute_costs
from src.forecaster     import forecast_dept


# ── cost columns we want to validate end-to-end ───────────────────────────
ACTUAL_COST_COLS = {
    "Direct Salary Cost (INR)":        "direct_salary_cost",
    "Benefits Cost (INR)":             "benefits_cost",
    "Variable Pay Bonus (INR)":        "variable_pay_bonus",
    "Payroll Tax (INR)":               "payroll_tax",
    "Travel Allowance (INR)":          "travel_allowance",
    "Meal Allowance (INR)":            "meal_allowance",
    "Overtime Cost (INR)":             "overtime_cost",
    "Recruitment Cost (INR)":          "recruitment_cost",
    "Training & Dev Cost (INR)":       "training_dev_cost",
    "IT License Cost (INR)":           "it_license_cost",
    "IT Equipment Cost (INR)":         "it_equipment_cost",
    "Rent & Facilities Cost (INR)":    "rent_facilities_cost",
    "Utilities Cost (INR)":            "utilities_cost",
    "Admin Overhead (INR)":            "admin_overhead",
    "HR Payroll Admin (INR)":          "hr_payroll_admin",
    "Learning Platform Cost (INR)":    "learning_platform_cost",
    "Employee Engagement Cost (INR)":  "employee_engagement_cost",
    "Other Indirect Cost (INR)":       "other_indirect_cost",
    "Total Direct Cost (INR)":         "total_direct_cost",
    "Total Indirect Cost (INR)":       "total_indirect_cost",
    "Total Actual Cost (INR)":         "total_cost",
}


# ── metric helpers ────────────────────────────────────────────────────────

def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)

def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))

def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))

def wmape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(actual)))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(actual - pred)) / denom * 100)

def dir_acc(actual: np.ndarray, pred: np.ndarray) -> float:
    if len(actual) < 2:
        return np.nan
    return float(np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(pred))) * 100)

def max_err(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.max(np.abs(actual - pred)))

def metrics_dict(actual: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "MAPE %":   round(mape(actual, pred), 2),
        "WMAPE %":  round(wmape(actual, pred), 2),
        "MAE":      round(mae(actual, pred), 0),
        "RMSE":     round(rmse(actual, pred), 0),
        "DirAcc %": round(dir_acc(actual, pred), 1),
        "MaxErr":   round(max_err(actual, pred), 0),
    }


def _seasonal_naive_from_train(
    train_series: pd.Series,
    test_index: pd.Index,
    season_len: int = 12,
) -> np.ndarray:
    """
    Seasonal naive forecast using only train history.
    For each test month t, predict y(t-season_len); if not available, use last train value.
    """
    if train_series.empty:
        return np.array([])
    hist = train_series.astype(float).copy()
    preds: list[float] = []
    last_val = float(hist.iloc[-1])
    for idx in test_index:
        try:
            lag_idx = idx - season_len
        except Exception:
            lag_idx = None
        if lag_idx is not None and lag_idx in hist.index:
            preds.append(float(hist.loc[lag_idx]))
        else:
            preds.append(last_val)
    return np.array(preds, dtype=float)


def _attach_naive_comparison(metrics: dict, actual: np.ndarray, naive_pred: np.ndarray) -> None:
    if len(actual) == 0 or len(naive_pred) == 0:
        return
    n = min(len(actual), len(naive_pred))
    base_mape = round(mape(actual[:n], naive_pred[:n]), 2)
    base_wmape = round(wmape(actual[:n], naive_pred[:n]), 2)
    metrics["Naive MAPE %"] = base_mape
    metrics["Naive WMAPE %"] = base_wmape
    # MAPE can explode when actual values are near zero; suppress non-informative lift values.
    if np.isfinite(base_mape) and np.isfinite(metrics["MAPE %"]) and base_mape <= 500 and metrics["MAPE %"] <= 500:
        metrics["MAPE Lift vs Naive %pts"] = round(base_mape - metrics["MAPE %"], 2)
    else:
        metrics["MAPE Lift vs Naive %pts"] = np.nan
    metrics["WMAPE Lift vs Naive %pts"] = round(base_wmape - metrics["WMAPE %"], 2)


# ── SARIMA series validation ───────────────────────────────────────────────

def validate_sarima_series(
    df: pd.DataFrame,
    train_end: str,
    test_start: str,
    test_end: str,
) -> dict[str, dict]:
    """
    Retrain SARIMA from scratch on [start:train_end], forecast to test window.
    Returns metrics dict per series.
    """
    train = df.loc[:train_end]
    test  = df.loc[test_start:test_end]
    results = {}

    active_map = _resolve_series_map(df)
    for key, col in active_map.items():
        if col not in df.columns:
            continue
        tr = train[col].dropna().astype(float)
        te = test[col].dropna().astype(float)
        if len(tr) < 12 or len(te) == 0:
            continue
        h = len(te)

        exog_tr = exog_te = None
        if key == "engagement":
            hc_col = "Closing HC"
            if hc_col in df.columns:
                exog_tr = train.loc[tr.index, hc_col].ffill().astype(float)
                exog_te = test.loc[te.index, hc_col].ffill().astype(float)

        try:
            fit, best_order, best_seasonal, best_aic = fit_best_sarima(tr, key, exog=exog_tr)
            fc  = fit.forecast(steps=h, exog=exog_te if exog_te is not None else None)
            actual_arr = te.values
            pred_arr   = fc.values[:len(actual_arr)]
            m = metrics_dict(actual_arr, pred_arr)
            naive_pred = _seasonal_naive_from_train(tr, te.index)
            _attach_naive_comparison(m, actual_arr, naive_pred)
            m["actual_mean"]    = round(float(actual_arr.mean()), 2)
            m["predicted_mean"] = round(float(pred_arr.mean()), 2)
            m["n_months"]       = len(actual_arr)
            m["selected_order"] = str(best_order)
            m["selected_seasonal_order"] = str(best_seasonal)
            m["AIC"] = round(best_aic, 2) if best_aic is not None else None
            results[key] = m
        except Exception as e:
            results[key] = {"error": str(e)}

    return results


def _fit_models_for_window(train: pd.DataFrame) -> tuple[dict[str, Any], dict]:
    """
    Fit the 5 SARIMA/SARIMAX models using only the given train window.
    Used by full-pipeline backtest to avoid leakage from future months.
    """
    models: dict[str, Any] = {}
    active_map = _resolve_series_map(train)
    for key, col in active_map.items():
        if col not in train.columns:
            continue
        tr = train[col].dropna().astype(float)
        if len(tr) < 12:
            continue

        exog_tr = None
        if key == "engagement" and "Closing HC" in train.columns:
            exog_tr = train.loc[tr.index, "Closing HC"].ffill().astype(float)

        try:
            fit, _, _, _ = fit_best_sarima(tr, key, exog=exog_tr)
            models[key] = fit
        except Exception:
            continue
    strategy = learn_series_strategy(train, min_lift_wmape=1.0, holdout_months=6)
    return models, strategy


# ── End-to-end cost validation ─────────────────────────────────────────────

def validate_end_to_end(
    df: pd.DataFrame,
    train_end: str,
    test_start: str,
    test_end: str,
) -> dict[str, dict]:
    """
    Uses actual HC & attrition values from the test window as inputs to the
    rules engine (bypassing SARIMA), so we isolate rules-engine accuracy
    from SARIMA accuracy.  Also runs full SARIMA forecast for comparison.
    """
    train = df.loc[:train_end]
    test  = df.loc[test_start:test_end]

    rates = learn_rates(train, lookback_months=18)
    results = {}

    # ── Rules-only accuracy (uses actual HC/attrition) ─────────────────
    rules_rows = []
    for period, row in test.iterrows():
        closing_hc   = float(row.get("Closing HC", 0))
        opening_hc   = float(row.get("Opening HC", closing_hc))
        attrition    = float(row.get("Attrition % Annualized", 0))
        avg_salary   = float(row.get("Avg Salary Per FTE Annual (INR)", 0))
        engagement   = float(row.get("Employee Engagement Cost (INR)", 0))
        other_ind    = float(row.get("Other Indirect Cost (INR)", 0))
        month        = period.month

        cost = compute_costs(
            closing_hc=closing_hc,
            opening_hc=opening_hc,
            attrition_pct=attrition,
            avg_salary_annual=avg_salary,
            month=month,
            rates=rates,
            engagement_override=engagement,
            other_indirect_override=other_ind,
        )
        cost["period"] = period
        rules_rows.append(cost)

    if not rules_rows:
        return results

    rules_df = pd.DataFrame(rules_rows).set_index("period")

    for actual_col, pred_col in ACTUAL_COST_COLS.items():
        if actual_col not in test.columns or pred_col not in rules_df.columns:
            continue
        actual_arr = test[actual_col].fillna(0).astype(float).values
        pred_arr   = rules_df[pred_col].fillna(0).astype(float).values
        h = min(len(actual_arr), len(pred_arr))
        m = metrics_dict(actual_arr[:h], pred_arr[:h])
        naive_pred = _seasonal_naive_from_train(train[actual_col].dropna().astype(float), test.index[:h])
        _attach_naive_comparison(m, actual_arr[:h], naive_pred[:h])
        m["actual_mean"]    = round(float(actual_arr[:h].mean()), 0)
        m["predicted_mean"] = round(float(pred_arr[:h].mean()), 0)
        results[actual_col] = m

    return results


def validate_full_pipeline(
    df: pd.DataFrame,
    train_end: str,
    test_start: str,
    test_end: str,
) -> dict[str, dict]:
    """
    True end-to-end backtest:
      1) fit SARIMA/SARIMAX on train window only
      2) forecast drivers for test horizon
      3) generate costs via rules engine from forecasted drivers
      4) compare to actual test costs
    """
    train = df.loc[:train_end]
    test  = df.loc[test_start:test_end]
    h = len(test)
    if len(train) < 12 or h == 0:
        return {}

    rates = learn_rates(train, lookback_months=18)
    models, strategy = _fit_models_for_window(train)
    if not models:
        return {}

    fc_df = forecast_dept(
        dept="Backtest",
        dept_df=train,
        models=models,
        rates=rates,
        horizon=h,
        override={},
        strategy=strategy,
    )
    fc_df = fc_df.iloc[:h].copy()
    fc_df.index = test.index[:len(fc_df)]

    results: dict[str, dict] = {}
    for actual_col, pred_col in ACTUAL_COST_COLS.items():
        if actual_col not in test.columns or pred_col not in fc_df.columns:
            continue
        actual_arr = test[actual_col].fillna(0).astype(float).values
        pred_arr   = fc_df[pred_col].fillna(0).astype(float).values
        n = min(len(actual_arr), len(pred_arr))
        if n == 0:
            continue
        m = metrics_dict(actual_arr[:n], pred_arr[:n])
        naive_pred = _seasonal_naive_from_train(train[actual_col].dropna().astype(float), test.index[:n])
        _attach_naive_comparison(m, actual_arr[:n], naive_pred[:n])
        m["actual_mean"]    = round(float(actual_arr[:n].mean()), 0)
        m["predicted_mean"] = round(float(pred_arr[:n].mean()), 0)
        results[actual_col] = m

    return results


# ── month-by-month comparison table ───────────────────────────────────────

def month_comparison_table(
    df: pd.DataFrame,
    train_end: str,
    test_start: str,
    test_end: str,
) -> pd.DataFrame:
    """
    Returns a detailed month-by-month table:
      Period | Actual Total Cost | Predicted Total Cost | Diff | Diff %
    Uses actual HC/attrition as inputs (rules-engine isolation test).
    """
    train = df.loc[:train_end]
    test  = df.loc[test_start:test_end]
    rates = learn_rates(train, lookback_months=18)

    rows = []
    for period, row in test.iterrows():
        closing_hc  = float(row.get("Closing HC", 0))
        opening_hc  = float(row.get("Opening HC", closing_hc))
        attrition   = float(row.get("Attrition % Annualized", 0))
        avg_salary  = float(row.get("Avg Salary Per FTE Annual (INR)", 0))
        engagement  = float(row.get("Employee Engagement Cost (INR)", 0))
        other_ind   = float(row.get("Other Indirect Cost (INR)", 0))
        month       = period.month

        cost = compute_costs(
            closing_hc=closing_hc, opening_hc=opening_hc,
            attrition_pct=attrition, avg_salary_annual=avg_salary,
            month=month, rates=rates,
            engagement_override=engagement, other_indirect_override=other_ind,
        )

        actual_total = float(row.get("Total Actual Cost (INR)", 0))
        pred_total   = cost["total_cost"]
        diff         = pred_total - actual_total
        diff_pct     = (diff / actual_total * 100) if actual_total else 0

        rows.append({
            "Period":             str(period),
            "Actual HC":          int(closing_hc),
            "Actual Total (INR)": round(actual_total, 0),
            "Pred Total (INR)":   round(pred_total, 0),
            "Diff (INR)":         round(diff, 0),
            "Diff %":             round(diff_pct, 2),
            "Actual Direct":      round(float(row.get("Total Direct Cost (INR)", 0)), 0),
            "Pred Direct":        round(cost["total_direct_cost"], 0),
            "Actual Indirect":    round(float(row.get("Total Indirect Cost (INR)", 0)), 0),
            "Pred Indirect":      round(cost["total_indirect_cost"], 0),
        })

    if not rows:
        return pd.DataFrame(
            columns=[
                "Actual HC",
                "Actual Total (INR)",
                "Pred Total (INR)",
                "Diff (INR)",
                "Diff %",
                "Actual Direct",
                "Pred Direct",
                "Actual Indirect",
                "Pred Indirect",
            ]
        )
    return pd.DataFrame(rows).set_index("Period")


# ── pretty print helpers ───────────────────────────────────────────────────

SEP = "─" * 110

def _inr(v):
    if pd.isna(v):
        return "N/A"
    v = float(v)
    sign = "-" if v < 0 else ""
    av = abs(v)
    if av >= 1e7:
        return f"{sign}₹{av/1e7:.2f} Cr"
    elif av >= 1e5:
        return f"{sign}₹{av/1e5:.2f} L"
    return f"{sign}₹{av:,.0f}"

def print_section(title: str):
    print(f"\n{'='*110}")
    print(f"  {title}")
    print('='*110)


def _p(v) -> str:
    if pd.isna(v):
        return "  N/A"
    return f"{float(v):6.1f}%"


# ── main ──────────────────────────────────────────────────────────────────

def run_evaluation(
    depts_filter: list[str] | None = None,
    rounds: list[str] | None = None,
    output_csv: str | None = None,
    data_path: Path | None = None,
):
    depts_filter = depts_filter or DEPARTMENTS
    rounds       = rounds or ["test25", "val26"]

    SPLITS = {
        "test25": ("2024-12", "2025-01", "2025-12"),
        "val26":  ("2025-12", "2026-01", "2026-12"),
    }

    print("\n" + "█"*110)
    print("  WORKFORCE COST FORECASTING — VALIDATION REPORT")
    print("  Data: Jan-2022 onward  |  Holdout windows: Test-2025, Validation-2026")
    print("█"*110)

    all_rows = []
    data = load_all(data_path or _default_data_path(BASE_DIR))

    for dept in depts_filter:
        if dept not in data:
            print(f"\n[WARN] Department '{dept}' not found — skipping.")
            continue
        df = data[dept]

        print_section(f"DEPARTMENT: {dept}")

        for rnd_key in rounds:
            if rnd_key not in SPLITS:
                print(f"  [WARN] Unknown round '{rnd_key}'. Valid: test25, val26")
                continue
            train_end, test_start, test_end = SPLITS[rnd_key]
            label = {
                "test25": "Test 2025 (Train 2022-2024)",
                "val26": "Validation 2026 (Train 2022-2025)",
            }[rnd_key]
            test_window = df.loc[test_start:test_end]
            if test_window.empty:
                print(f"\n  ┌── {label} ──────────────────────────────────────────────────────────────────┐")
                print(f"  [WARN] No rows available in {test_start}→{test_end} for {dept}; skipping this round.")
                print(f"  └{'─'*108}┘")
                continue

            print(f"\n  ┌── {label} ──────────────────────────────────────────────────────────────────┐")

            # ── Month-by-month comparison ──────────────────────────────
            cmp = month_comparison_table(df, train_end, test_start, test_end)
            print(f"\n  Month-by-Month (Total Cost) — rules engine with ACTUAL HC/attrition inputs")
            print(f"  {'Period':<10} {'Actual Total':>16} {'Pred Total':>16} {'Diff':>16} {'Diff %':>8} "
                  f"{'Act Direct':>15} {'Pred Direct':>15} {'Act Indirect':>14} {'Pred Indirect':>14}")
            print("  " + SEP)
            for period, row in cmp.iterrows():
                flag = "⚠" if abs(row["Diff %"]) > 10 else "✓"
                print(f"  {period:<10} {_inr(row['Actual Total (INR)']):>16} {_inr(row['Pred Total (INR)']):>16} "
                      f"{_inr(row['Diff (INR)']):>16} {row['Diff %']:>7.1f}% {flag}  "
                      f"{_inr(row['Actual Direct']):>14} {_inr(row['Pred Direct']):>14} "
                      f"{_inr(row['Actual Indirect']):>13} {_inr(row['Pred Indirect']):>14}")

            # ── SARIMA series metrics ──────────────────────────────────
            s_metrics = validate_sarima_series(df, train_end, test_start, test_end)
            print(f"\n  SARIMA Series Accuracy")
            print(f"  {'Series':<20} {'MAPE':>8} {'WMAPE':>8} {'Lift':>8} {'MAE':>14} {'RMSE':>14} {'Dir':>7} {'MaxErr':>14} {'Actual μ':>14} {'Pred μ':>14}")
            print("  " + SEP)
            for series, m in s_metrics.items():
                if "error" in m:
                    print(f"  {series:<22}  ERROR: {m['error']}")
                    continue
                print(
                    f"  {series:<20} {_p(m.get('MAPE %')):>8} {_p(m.get('WMAPE %')):>8} "
                    f"{m.get('MAPE Lift vs Naive %pts', np.nan):>7.1f} "
                    f"{_inr(m['MAE']):>14} {_inr(m['RMSE']):>14} "
                    f"{m['DirAcc %']:>6.0f}% {_inr(m['MaxErr']):>14} "
                    f"{_inr(m['actual_mean']):>14} {_inr(m['predicted_mean']):>14}"
                )

            # ── Cost-line metrics (rules-isolation) ───────────────────
            e_metrics = validate_end_to_end(df, train_end, test_start, test_end)
            print(f"\n  Cost Line Accuracy (rules-only: ACTUAL driver inputs)")
            print(f"  {'Cost Line':<34} {'MAPE':>8} {'WMAPE':>8} {'Lift':>8} {'MAE':>14} {'RMSE':>14} {'Dir':>7} {'Actual μ':>14} {'Pred μ':>14}")
            print("  " + SEP)
            for col, m in e_metrics.items():
                if "error" in m:
                    print(f"  {col:<38}  ERROR: {m['error']}")
                    continue
                flag = " ⚠" if m["MAPE %"] > 20 else ""
                print(
                    f"  {col:<34} {_p(m.get('MAPE %')):>8}{flag:<2} {_p(m.get('WMAPE %')):>8} "
                    f"{m.get('MAPE Lift vs Naive %pts', np.nan):>7.1f} "
                    f"{_inr(m['MAE']):>14} {_inr(m['RMSE']):>14} "
                    f"{m['DirAcc %']:>6.0f}% {_inr(m['actual_mean']):>14} {_inr(m['predicted_mean']):>14}"
                )

            # ── Cost-line metrics (true forecasting) ───────────────────
            fp_metrics = validate_full_pipeline(df, train_end, test_start, test_end)
            print(f"\n  Cost Line Accuracy (full pipeline: FORECASTED drivers)")
            print(f"  {'Cost Line':<34} {'MAPE':>8} {'WMAPE':>8} {'Lift':>8} {'MAE':>14} {'RMSE':>14} {'Dir':>7} {'Actual μ':>14} {'Pred μ':>14}")
            print("  " + SEP)
            for col, m in fp_metrics.items():
                if "error" in m:
                    print(f"  {col:<38}  ERROR: {m['error']}")
                    continue
                flag = " ⚠" if m["MAPE %"] > 20 else ""
                print(
                    f"  {col:<34} {_p(m.get('MAPE %')):>8}{flag:<2} {_p(m.get('WMAPE %')):>8} "
                    f"{m.get('MAPE Lift vs Naive %pts', np.nan):>7.1f} "
                    f"{_inr(m['MAE']):>14} {_inr(m['RMSE']):>14} "
                    f"{m['DirAcc %']:>6.0f}% {_inr(m['actual_mean']):>14} {_inr(m['predicted_mean']):>14}"
                )

            print(f"\n  └{'─'*108}┘")

            # Collect CSV rows
            for col, m in e_metrics.items():
                if "error" in m:
                    continue
                all_rows.append({
                    "Dept": dept, "Round": label, "Cost Line": col,
                    "Evaluation Type": "Rules-only (actual drivers)",
                    **m
                })
            for col, m in fp_metrics.items():
                if "error" in m:
                    continue
                all_rows.append({
                    "Dept": dept, "Round": label, "Cost Line": col,
                    "Evaluation Type": "Full pipeline (forecast drivers)",
                    **m
                })

        # ── Overall summary for this dept (validation year) ─────────────────
        if "val26" in rounds:
            train_end, test_start, test_end = SPLITS["val26"]
            if not df.loc[test_start:test_end].empty:
                e_metrics = validate_end_to_end(df, train_end, test_start, test_end)
                total_metrics = e_metrics.get("Total Actual Cost (INR)", {})
                if total_metrics and "error" not in total_metrics:
                    print(f"\n  ★ OVERALL (Validation 2026) — Total Cost:")
                    print(f"    MAPE={total_metrics['MAPE %']:.1f}%   "
                          f"WMAPE={total_metrics['WMAPE %']:.1f}%   "
                          f"MAE={_inr(total_metrics['MAE'])}   "
                          f"RMSE={_inr(total_metrics['RMSE'])}   "
                          f"DirAcc={total_metrics['DirAcc %']:.0f}%   "
                          f"MaxErr={_inr(total_metrics['MaxErr'])}")
                fp_metrics = validate_full_pipeline(df, train_end, test_start, test_end)
                fp_total = fp_metrics.get("Total Actual Cost (INR)", {})
                if fp_total and "error" not in fp_total:
                    print(f"    Full pipeline (forecast drivers) → "
                          f"MAPE={fp_total['MAPE %']:.1f}%   "
                          f"WMAPE={fp_total['WMAPE %']:.1f}%   "
                          f"LiftVsNaive={fp_total.get('MAPE Lift vs Naive %pts', np.nan):.1f} pts   "
                          f"MAE={_inr(fp_total['MAE'])}   "
                          f"RMSE={_inr(fp_total['RMSE'])}   "
                          f"DirAcc={fp_total['DirAcc %']:.0f}%   "
                          f"MaxErr={_inr(fp_total['MaxErr'])}")

    # ── Save CSV ───────────────────────────────────────────────────────────
    if output_csv and all_rows:
        out = Path(output_csv)
        pd.DataFrame(all_rows).to_csv(out, index=False)
        print(f"\n\nMetrics saved → {out.resolve()}")

    print("\n" + "█"*110)
    print("  END OF VALIDATION REPORT")
    print("█"*110 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate workforce cost forecast model.")
    parser.add_argument(
        "--dept", nargs="+", default=None,
        help="One or more department names (default: all 6)",
    )
    parser.add_argument(
        "--rounds", nargs="+", default=["test25", "val26"],
        help="Validation rounds to run: test25  val26  (default: both)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save CSV metrics report (optional)",
    )
    parser.add_argument(
        "--data",
        default=str(_default_data_path(BASE_DIR)),
        help="Path to source Excel workbook.",
    )
    args = parser.parse_args()
    run_evaluation(
        depts_filter=args.dept,
        rounds=args.rounds,
        output_csv=args.output,
        data_path=Path(args.data),
    )
