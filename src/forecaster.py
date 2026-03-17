"""
forecaster.py
=============
Core forecasting orchestration:
  - Forecast model-led drivers with SARIMA/SARIMAX (or seasonal naive by strategy)
  - Apply deterministic formula engine for cost construction
  - Keep optional user overrides for simulation
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

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

EXOG_DRIVER_KEYS: dict[str, list[str]] = {
    "monthly_budget": ["closing_hc"],
    "attrition_pct": ["closing_hc"],
    "engagement": ["closing_hc", "monthly_budget"],
    "other_indirect": ["closing_hc", "monthly_budget"],
}


def _sarima_forecast(result, horizon: int, exog=None) -> np.ndarray:
    try:
        fc = result.forecast(steps=horizon, exog=exog)
        return np.maximum(0.0, fc.values)
    except Exception:
        return np.array([float(result.fittedvalues.iloc[-1])] * horizon)


def _forecast_model_series(result, horizon: int, exog: np.ndarray | None = None) -> np.ndarray:
    k_exog = int(getattr(getattr(result, "model", None), "k_exog", 0) or 0)
    if k_exog <= 0:
        return _sarima_forecast(result, horizon)
    if exog is None:
        return _sarima_forecast(result, horizon)
    arr = np.asarray(exog, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.shape[1] > k_exog:
        arr = arr[:, :k_exog]
    elif arr.shape[1] < k_exog:
        pad = np.zeros((arr.shape[0], k_exog - arr.shape[1]), dtype=float)
        arr = np.hstack([arr, pad])
    return _sarima_forecast(result, horizon, exog=arr)


def _seasonal_naive_forecast(series: pd.Series, periods: pd.PeriodIndex, season_len: int = 12) -> np.ndarray:
    if series.empty:
        return np.array([np.nan] * len(periods), dtype=float)
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


def _rule_based_budget_forecast(dept_df: pd.DataFrame, periods: pd.PeriodIndex) -> np.ndarray:
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


def forecast_dept(
    dept: str,
    dept_df: pd.DataFrame,
    models: dict,
    rates: dict,
    horizon: int = 12,
    override: dict | None = None,
    strategy: dict | None = None,
) -> pd.DataFrame:
    override = override or {}
    strategy = strategy or {}

    last_period = dept_df.index[-1]
    periods = pd.period_range(last_period + 1, periods=horizon, freq="M")

    def _strategy_for(key: str) -> str:
        cfg = strategy.get(key)
        if isinstance(cfg, dict):
            return str(cfg.get("strategy", "sarima")).lower()
        if isinstance(cfg, str):
            return cfg.lower()
        return "sarima"

    def _series_for_key(key: str) -> pd.Series:
        col = DRIVER_COL_MAP.get(key)
        if key in BAND_HC_FALLBACK_COLS:
            col = next((c for c in BAND_HC_FALLBACK_COLS[key] if c in dept_df.columns), col)
        if col and col in dept_df.columns:
            return dept_df[col].dropna().astype(float)
        return pd.Series(dtype=float)

    def _get_fc(key: str, n: int = horizon) -> np.ndarray:
        method = _strategy_for(key)
        if method == "naive":
            return _seasonal_naive_forecast(_series_for_key(key), periods[:n])
        if key in models:
            return _forecast_model_series(models[key], n)
        series = _series_for_key(key)
        if not series.empty:
            return _seasonal_naive_forecast(series, periods[:n])
        return np.array([np.nan] * n, dtype=float)

    def _exog_from_fc(keys: list[str], fc_map: dict[str, np.ndarray]) -> np.ndarray | None:
        cols = []
        for k in keys:
            v = fc_map.get(k)
            if v is None:
                return None
            cols.append(np.asarray(v, dtype=float).reshape(-1, 1))
        return np.hstack(cols) if cols else None

    hc_fc = _get_fc("closing_hc")

    budget_fc = _get_fc("monthly_budget")
    if _strategy_for("monthly_budget") != "naive" and "monthly_budget" in models:
        mb_exog = _exog_from_fc(EXOG_DRIVER_KEYS.get("monthly_budget", []), {"closing_hc": hc_fc})
        budget_fc = _forecast_model_series(models["monthly_budget"], horizon, exog=mb_exog)

    attrition_fc = _get_fc("attrition_pct")
    if _strategy_for("attrition_pct") != "naive" and "attrition_pct" in models:
        at_exog = _exog_from_fc(EXOG_DRIVER_KEYS.get("attrition_pct", []), {"closing_hc": hc_fc})
        attrition_fc = _forecast_model_series(models["attrition_pct"], horizon, exog=at_exog)

    salary_fc = _get_fc("avg_salary")
    engagement_fc = _get_fc("engagement")
    other_fc = _get_fc("other_indirect")
    band_fc = {k: _get_fc(k) for k in BAND_HC_KEYS}

    # SARIMAX exogenous coupling for engagement and other indirect (HC-linked).
    if _strategy_for("engagement") != "naive" and "engagement" in models:
        try:
            eng_exog = _exog_from_fc(
                EXOG_DRIVER_KEYS.get("engagement", []),
                {"closing_hc": hc_fc, "monthly_budget": budget_fc},
            )
            engagement_fc = _forecast_model_series(models["engagement"], horizon, exog=eng_exog)
        except Exception:
            pass
    if _strategy_for("other_indirect") != "naive" and "other_indirect" in models:
        try:
            oi_exog = _exog_from_fc(
                EXOG_DRIVER_KEYS.get("other_indirect", []),
                {"closing_hc": hc_fc, "monthly_budget": budget_fc},
            )
            other_fc = _forecast_model_series(models["other_indirect"], horizon, exog=oi_exog)
        except Exception:
            pass

    def _apply_override(base: np.ndarray, key: str) -> np.ndarray:
        val = override.get(key)
        if val is None:
            return base
        arr = _to_override_array(val, len(base))
        if arr is not None:
            return arr
        return np.full(len(base), float(val))

    hc_fc = _apply_override(hc_fc, "closing_hc")
    attrition_fc = _apply_override(attrition_fc, "attrition_pct")
    salary_fc = _apply_override(salary_fc, "avg_salary")
    engagement_fc = _apply_override(engagement_fc, "engagement")
    other_fc = _apply_override(other_fc, "other_indirect")
    budget_fc = _apply_override(budget_fc, "monthly_budget")
    for k in BAND_HC_KEYS:
        band_fc[k] = _apply_override(band_fc[k], k)

    # Budget fallback stays deterministic (no manual patching).
    if override.get("monthly_budget") is None and np.isnan(budget_fc).all():
        budget_fc = _rule_based_budget_forecast(dept_df, periods)
    budget_fc = np.maximum(0.0, _safe_num(budget_fc))

    # Only link band-HC to total HC when user explicitly overrides band values.
    link_band_hc = bool(override.get("link_band_hc_to_closing_hc", True))
    band_override_present = any(k in override and override.get(k) is not None for k in BAND_HC_KEYS)
    if link_band_hc and band_override_present:
        band_matrix = np.vstack([band_fc[k] for k in BAND_HC_KEYS])
        valid_band_month = np.any(~np.isnan(band_matrix), axis=0)
        band_sum = np.nansum(band_matrix, axis=0)
        hc_fc = np.where(valid_band_month, np.maximum(1.0, band_sum), hc_fc)

    inc_pct = override.get("increment_pct")
    inc_arr = _to_override_array(inc_pct, horizon)
    if inc_arr is not None:
        salary_fc = salary_fc * (1.0 + inc_arr)
    elif inc_pct is not None:
        salary_fc = salary_fc * (1.0 + float(inc_pct))

    wfh_override = override.get("wfh_pct", rates.get("wfh_pct", 0.55))

    rows = []
    opening_hc = float(dept_df["Closing HC"].iloc[-1])

    for i, period in enumerate(periods):
        c_hc = max(1.0, float(hc_fc[i]))
        att_pct = max(0.0, float(attrition_fc[i]))
        salary = max(0.0, float(salary_fc[i]))
        wfh_pct_i = min(1.0, max(0.0, _step_value(wfh_override, i, rates.get("wfh_pct", 0.55))))

        cost_dict = compute_costs(
            closing_hc=c_hc,
            opening_hc=opening_hc,
            attrition_pct=att_pct,
            avg_salary_annual=salary,
            month=int(period.month),
            rates=rates,
            wfh_pct=wfh_pct_i,
            engagement_override=float(engagement_fc[i]) if not np.isnan(engagement_fc[i]) else None,
            other_indirect_override=float(other_fc[i]) if not np.isnan(other_fc[i]) else None,
        )

        cost_dict["period"] = str(period)
        cost_dict["attrition_pct"] = att_pct
        cost_dict["avg_salary"] = salary
        if i < len(budget_fc) and not np.isnan(budget_fc[i]):
            cost_dict["monthly_budget"] = float(budget_fc[i])

        band_total = 0.0
        has_band = False
        for key in BAND_HC_KEYS:
            arr = band_fc.get(key)
            if arr is None or i >= len(arr):
                continue
            v = float(arr[i])
            if np.isnan(v):
                continue
            vv = max(0.0, v)
            cost_dict[key] = vv
            band_total += vv
            has_band = True
        if has_band:
            cost_dict["band_hc_total"] = band_total

        rows.append(cost_dict)
        opening_hc = c_hc

    result_df = pd.DataFrame(rows)
    result_df["period"] = pd.PeriodIndex(result_df["period"], freq="M")
    return result_df.set_index("period")


def extrapolate_annual_budget(
    dept_df: pd.DataFrame,
    target_fy_start: str = "2026-01",
) -> float:
    """
    Robust annual budget extrapolation with deterministic fallbacks.
    """
    fy_col = "FY Annual Budget Pool (INR)"
    if fy_col in dept_df.columns:
        jan_rows = dept_df[dept_df.index.month == 1]
        budgets = jan_rows[fy_col].dropna().astype(float)
        if len(budgets) >= 2:
            years = np.arange(len(budgets))
            vals = budgets.values.astype(float)
            m, b = np.polyfit(years, vals, 1)
            return max(0.0, float(m * len(years) + b))
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
            return max(0.0, float(ob.iloc[-1] * 12.0))

    total_col = "Total Actual Cost (INR)"
    if total_col in dept_df.columns:
        actual = dept_df[total_col].dropna().astype(float)
        if len(actual) >= 12:
            return max(0.0, float(actual.tail(12).sum()))
        if len(actual) > 0:
            return max(0.0, float(actual.iloc[-1] * 12.0))

    return 0.0
