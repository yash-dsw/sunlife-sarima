"""
forecaster.py
=============
Orchestrates SARIMA models + cost rules engine to produce a 12-month
monthly forecast for a single department.

Public API
----------
forecast_dept(dept, depts_data, models, rates, horizon=12, override=None)
  → pd.DataFrame  (one row per forecast month)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any

from src.cost_rules import compute_costs

DRIVER_COL_MAP = {
    "closing_hc": "Closing HC",
    "attrition_pct": "Attrition % Annualized",
    "avg_salary": "Avg Salary Per FTE Annual (INR)",
    "engagement": "Employee Engagement Cost (INR)",
    "other_indirect": "Other Indirect Cost (INR)",
    "monthly_budget": "Original Budget (INR)",
    "band_hc_1": "Band 1 Count",
    "band_hc_2": "Band 2 Count",
    "band_hc_3": "Band 3 Count",
    "band_hc_4": "Band 4 Count",
    "band_hc_5": "Band 5 Count",
    "band_hc_6": "Band 6 Count",
}

BAND_HC_KEYS = ["band_hc_1", "band_hc_2", "band_hc_3", "band_hc_4", "band_hc_5", "band_hc_6"]

BAND_HC_FALLBACK_COLS = {
    "band_hc_1": ["Band 1 Count", "Band A Count"],
    "band_hc_2": ["Band 2 Count", "Band B Count"],
    "band_hc_3": ["Band 3 Count", "Band C Count"],
    "band_hc_4": ["Band 4 Count", "Band D Count"],
    "band_hc_5": ["Band 5 Count", "Band E Count"],
    "band_hc_6": ["Band 6 Count", "Band F Count"],
}


def _sarima_forecast(result, horizon: int, exog=None) -> np.ndarray:
    """Return numpy array of length `horizon` from a fitted SARIMAX result."""
    try:
        fc = result.forecast(steps=horizon, exog=exog)
        return np.maximum(0.0, fc.values)
    except Exception:
        return np.array([result.fittedvalues.iloc[-1]] * horizon)


def _seasonal_naive_forecast(series: pd.Series, periods: pd.PeriodIndex, season_len: int = 12) -> np.ndarray:
    if series.empty:
        return np.array([np.nan] * len(periods))
    hist = series.astype(float).copy()
    last_val = float(hist.iloc[-1])
    preds: list[float] = []
    for p in periods:
        lag_p = p - season_len
        if lag_p in hist.index:
            preds.append(float(hist.loc[lag_p]))
        else:
            preds.append(last_val)
    return np.array(preds, dtype=float)


def _safe_num(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr.astype(float), nan=0.0, posinf=0.0, neginf=0.0)


def _rule_based_budget_forecast(dept_df: pd.DataFrame, periods: pd.PeriodIndex) -> np.ndarray:
    """
    Deterministic fallback for monthly budget when model output is unavailable:
      1) Forecast annual pool using extrapolate_annual_budget()
      2) Allocate by historical month-wise seasonality weights
    """
    budget_col = "Original Budget (INR)"
    if budget_col not in dept_df.columns:
        return np.array([np.nan] * len(periods), dtype=float)
    hist = dept_df[budget_col].dropna().astype(float)
    if hist.empty:
        return np.array([np.nan] * len(periods), dtype=float)

    annual_pool = max(0.0, float(extrapolate_annual_budget(dept_df)))
    if annual_pool <= 0:
        return np.array([float(hist.iloc[-1])] * len(periods), dtype=float)

    month_mean = hist.groupby(hist.index.month).mean()
    default_w = float(month_mean.mean()) if len(month_mean) else 1.0
    w = np.array([float(month_mean.get(int(p.month), default_w)) for p in periods], dtype=float)
    w = np.maximum(0.0, w)
    if w.sum() <= 0:
        return np.array([annual_pool / max(len(periods), 1)] * len(periods), dtype=float)
    return annual_pool * (w / w.sum())


def _to_override_array(val: Any, n: int) -> np.ndarray | None:
    if val is None or isinstance(val, (str, bytes)):
        return None
    try:
        arr = np.asarray(val, dtype=float).reshape(-1)
    except Exception:
        return None
    if len(arr) != n:
        return None
    return arr


def _step_value(val: Any, i: int, default: float) -> float:
    if val is not None and not isinstance(val, (str, bytes)):
        try:
            arr = np.asarray(val, dtype=float).reshape(-1)
            if len(arr) > i:
                return float(arr[i])
        except Exception:
            pass
    try:
        return float(val)
    except Exception:
        return float(default)


def forecast_dept(
    dept: str,
    dept_df: pd.DataFrame,
    models: dict,
    rates: dict,
    horizon: int = 12,
    override: dict | None = None,
    strategy: dict | None = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    dept_df   : historical DataFrame for the department (from data_loader)
    models    : dict of fitted SARIMA results keyed by series name
    rates     : learned per-FTE/per-hire rates (from cost_rules.learn_rates)
    horizon   : number of months to forecast
    override  : optional dict of user tweaks, e.g.
                {
                  'closing_hc':    [130]*12,   # list of 12 monthly values
                  'attrition_pct': 0.25,       # scalar applied to all months
                  'avg_salary':    None,        # None = use SARIMA
                  'wfh_pct':       0.55,
                  'increment_pct': 0.10,       # applied on top of SARIMA salary
                }

    Returns
    -------
    pd.DataFrame indexed by Period (monthly), one column per cost metric.
    """
    override = override or {}
    strategy = strategy or {}
    last_period = dept_df.index[-1]
    periods = pd.period_range(last_period + 1, periods=horizon, freq="M")

    # ── Forecast the modelled driver series ─────────────────────────────
    def _strategy_for(key: str) -> str:
        cfg = strategy.get(key)
        if isinstance(cfg, dict):
            return str(cfg.get("strategy", "sarima")).lower()
        if isinstance(cfg, str):
            return cfg.lower()
        return "sarima"

    def _get_fc(key: str, n: int = horizon) -> np.ndarray:
        # Budget should follow model trend when model is available.
        # Seasonal-naive is only a fallback when no valid model exists.
        if key == "monthly_budget" and key in models:
            return _sarima_forecast(models[key], n)

        method = _strategy_for(key)
        if method == "naive":
            col = DRIVER_COL_MAP.get(key)
            if key in BAND_HC_FALLBACK_COLS:
                col = next((c for c in BAND_HC_FALLBACK_COLS[key] if c in dept_df.columns), col)
            if col and col in dept_df.columns:
                series = dept_df[col].dropna().astype(float)
                return _seasonal_naive_forecast(series, periods[:n])
        if key in models:
            return _sarima_forecast(models[key], n)
        return np.full(n, np.nan)

    hc_fc         = _get_fc("closing_hc")
    attrition_fc  = _get_fc("attrition_pct")
    salary_fc     = _get_fc("avg_salary")
    engagement_fc = _get_fc("engagement")
    other_fc      = _get_fc("other_indirect")
    budget_fc     = _get_fc("monthly_budget")
    # Keep a robust fallback for monthly budget trend control.
    budget_naive_fc = np.full(horizon, np.nan)
    budget_col = DRIVER_COL_MAP.get("monthly_budget")
    if budget_col and budget_col in dept_df.columns:
        budget_hist = dept_df[budget_col].dropna().astype(float)
        if not budget_hist.empty:
            budget_naive_fc = _seasonal_naive_forecast(budget_hist, periods)
    band_fc = {k: _get_fc(k) for k in BAND_HC_KEYS}

    # SARIMAX engagement needs HC exog
    if _strategy_for("engagement") != "naive" and "engagement" in models:
        try:
            engagement_fc = _sarima_forecast(
                models["engagement"], horizon, exog=hc_fc.reshape(-1, 1)
            )
        except Exception:
            pass

    # ── Apply overrides ─────────────────────────────────────────────────
    def _apply_override(base: np.ndarray, key: str) -> np.ndarray:
        val = override.get(key)
        if val is None:
            return base
        arr = _to_override_array(val, len(base))
        if arr is not None:
            return arr
        return np.full(len(base), float(val))

    hc_fc        = _apply_override(hc_fc,        "closing_hc")
    attrition_fc = _apply_override(attrition_fc, "attrition_pct")
    budget_fc    = _apply_override(budget_fc,    "monthly_budget")
    engagement_fc = _apply_override(engagement_fc, "engagement")
    other_fc      = _apply_override(other_fc, "other_indirect")
    for k in BAND_HC_KEYS:
        band_fc[k] = _apply_override(band_fc[k], k)
    wfh_override = override.get("wfh_pct", rates.get("wfh_pct", 0.55))

    budget_override_provided = override.get("monthly_budget") is not None
    if not budget_override_provided:
        # Pure model output only. If model has no budget series output, use deterministic rule-based fallback.
        if np.isnan(budget_fc).all():
            if not np.isnan(budget_naive_fc).all():
                budget_fc = budget_naive_fc.copy()
            else:
                budget_fc = _rule_based_budget_forecast(dept_df, periods)

    budget_fc = np.maximum(0.0, _safe_num(budget_fc))

    # Interdependency: if band HC is available or overridden, total HC follows band sum.
    link_band_hc = bool(override.get("link_band_hc_to_closing_hc", True))
    band_override_present = any(k in override and override.get(k) is not None for k in BAND_HC_KEYS)
    closing_hc_overridden = override.get("closing_hc") is not None
    band_matrix = np.vstack([band_fc[k] for k in BAND_HC_KEYS])
    valid_band_month = np.any(~np.isnan(band_matrix), axis=0)
    band_sum = np.nansum(band_matrix, axis=0)
    band_forecast_available = bool(np.any(valid_band_month))
    if link_band_hc and (band_override_present or (band_forecast_available and not closing_hc_overridden)):
        hc_fc = np.where(valid_band_month, np.maximum(1.0, band_sum), hc_fc)

    # increment override: scale the SARIMA salary forecast by (1 + inc)
    salary_override = override.get("avg_salary")
    inc_pct         = override.get("increment_pct")
    salary_override_arr = _to_override_array(salary_override, horizon)
    if salary_override_arr is not None:
        salary_fc = salary_override_arr
    elif salary_override is not None:
        salary_fc = np.full(horizon, float(salary_override))

    inc_arr = _to_override_array(inc_pct, horizon)
    if inc_arr is not None:
        salary_fc = salary_fc * (1.0 + inc_arr)
    elif inc_pct is not None:
        salary_fc = salary_fc * (1.0 + float(inc_pct))

    # ── Build monthly forecasts ─────────────────────────────────────────
    rows = []
    opening_hc = float(dept_df["Closing HC"].iloc[-1])

    for i, period in enumerate(periods):
        c_hc      = max(1.0, hc_fc[i])
        att_pct   = max(0.0, attrition_fc[i])
        salary    = max(0.0, salary_fc[i])
        wfh_pct_i = min(1.0, max(0.0, _step_value(wfh_override, i, rates.get("wfh_pct", 0.55))))
        month     = period.month

        cost_dict = compute_costs(
            closing_hc=c_hc,
            opening_hc=opening_hc,
            attrition_pct=att_pct,
            avg_salary_annual=salary,
            month=month,
            rates=rates,
            wfh_pct=wfh_pct_i,
            engagement_override=float(engagement_fc[i]) if not np.isnan(engagement_fc[i]) else None,
            other_indirect_override=float(other_fc[i])  if not np.isnan(other_fc[i])      else None,
        )
        cost_dict["period"]         = str(period)
        cost_dict["attrition_pct"]  = att_pct
        cost_dict["avg_salary"]     = salary
        if i < len(budget_fc) and not np.isnan(budget_fc[i]):
            cost_dict["monthly_budget"] = max(0.0, float(budget_fc[i]))
        band_total = 0.0
        has_band = False
        for key in BAND_HC_KEYS:
            arr = band_fc.get(key)
            if arr is not None and i < len(arr):
                v = float(arr[i])
                if not np.isnan(v):
                    vv = max(0.0, v)
                    cost_dict[key] = vv
                    band_total += vv
                    has_band = True
        if has_band:
            cost_dict["band_hc_total"] = band_total

        rows.append(cost_dict)
        opening_hc = c_hc   # roll forward

    result_df = pd.DataFrame(rows)
    result_df["period"] = pd.PeriodIndex(result_df["period"], freq="M")
    result_df = result_df.set_index("period")
    return result_df


def extrapolate_annual_budget(
    dept_df: pd.DataFrame,
    target_fy_start: str = "2026-01",
) -> float:
    """
    Estimate next annual budget with robust fallbacks.
    Priority:
      1) FY Annual Budget Pool (Jan rows), extrapolated when enough history exists
      2) Sum of last 12 months of Original Budget
      3) Sum of last 12 months of Total Actual Cost
      4) 0.0
    """
    fy_col = "FY Annual Budget Pool (INR)"
    if fy_col in dept_df.columns:
        jan_rows = dept_df[dept_df.index.month == 1]
        budgets = jan_rows[fy_col].dropna().astype(float)
        if len(budgets) >= 2:
            years = np.arange(len(budgets))
            vals = budgets.values.astype(float)
            m, b = np.polyfit(years, vals, 1)
            next_yr = len(years)
            return max(0.0, float(m * next_yr + b))
        if len(budgets) == 1:
            return max(0.0, float(budgets.iloc[-1]))

        any_budgets = dept_df[fy_col].dropna().astype(float)
        if len(any_budgets) > 0:
            return max(0.0, float(any_budgets.iloc[-1]))

    ob_col = "Original Budget (INR)"
    if ob_col in dept_df.columns:
        ob = dept_df[ob_col].dropna().astype(float)
        if len(ob) >= 12:
            return max(0.0, float(ob.tail(12).sum()))
        if len(ob) > 0:
            return max(0.0, float(ob.iloc[-1] * 12))

    total_col = "Total Actual Cost (INR)"
    if total_col in dept_df.columns:
        actual = dept_df[total_col].dropna().astype(float)
        if len(actual) >= 12:
            return max(0.0, float(actual.tail(12).sum()))
        if len(actual) > 0:
            return max(0.0, float(actual.iloc[-1] * 12))

    return 0.0
