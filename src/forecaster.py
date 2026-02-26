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

    # ── Forecast the 5 SARIMA series ────────────────────────────────────
    def _strategy_for(key: str) -> str:
        cfg = strategy.get(key)
        if isinstance(cfg, dict):
            return str(cfg.get("strategy", "sarima")).lower()
        if isinstance(cfg, str):
            return cfg.lower()
        return "sarima"

    def _get_fc(key: str, n: int = horizon) -> np.ndarray:
        method = _strategy_for(key)
        if method == "naive":
            col = DRIVER_COL_MAP.get(key)
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
        if hasattr(val, "__len__"):
            arr = np.asarray(val, dtype=float)
            if len(arr) == len(base):
                return arr
        return np.full(len(base), float(val))

    hc_fc        = _apply_override(hc_fc,        "closing_hc")
    attrition_fc = _apply_override(attrition_fc, "attrition_pct")

    # increment override: scale the SARIMA salary forecast by (1 + inc)
    salary_override = override.get("avg_salary")
    inc_pct         = override.get("increment_pct")
    if salary_override is not None:
        salary_fc = np.full(horizon, float(salary_override))
    elif inc_pct is not None:
        salary_fc = salary_fc * (1.0 + float(inc_pct))

    wfh_pct = override.get("wfh_pct", rates.get("wfh_pct", 0.55))

    # ── Build monthly forecasts ─────────────────────────────────────────
    rows = []
    opening_hc = float(dept_df["Closing HC"].iloc[-1])

    for i, period in enumerate(periods):
        c_hc      = max(1.0, hc_fc[i])
        att_pct   = max(0.0, attrition_fc[i])
        salary    = max(0.0, salary_fc[i])
        month     = period.month

        cost_dict = compute_costs(
            closing_hc=c_hc,
            opening_hc=opening_hc,
            attrition_pct=att_pct,
            avg_salary_annual=salary,
            month=month,
            rates=rates,
            wfh_pct=wfh_pct,
            engagement_override=float(engagement_fc[i]) if not np.isnan(engagement_fc[i]) else None,
            other_indirect_override=float(other_fc[i])  if not np.isnan(other_fc[i])      else None,
        )
        cost_dict["period"]         = str(period)
        cost_dict["attrition_pct"]  = att_pct
        cost_dict["avg_salary"]     = salary

        rows.append(cost_dict)
        opening_hc = c_hc   # roll forward

    result_df = pd.DataFrame(rows)
    result_df["period"] = pd.PeriodIndex(result_df["period"], freq="M")
    result_df = result_df.set_index("period")
    return result_df


def extrapolate_annual_budget(
    dept_df: pd.DataFrame,
    target_fy_start: str = "2025-04",
) -> float:
    """
    Simple linear extrapolation of 3-year FY budget pool to predict FY26.
    """
    april_rows = dept_df[dept_df.index.month == 4]
    budgets = april_rows["FY Annual Budget Pool (INR)"].dropna()
    if len(budgets) < 2:
        return float(dept_df["FY Annual Budget Pool (INR)"].dropna().iloc[-1])

    years = np.arange(len(budgets))
    vals  = budgets.values.astype(float)
    if len(years) >= 2:
        m, b = np.polyfit(years, vals, 1)
        next_yr = len(years)
        return max(0.0, float(m * next_yr + b))
    return float(vals[-1])
