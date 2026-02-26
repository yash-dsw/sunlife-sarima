"""
model_trainer.py
================
Trains 5 SARIMA models per department:
  1. Closing HC
  2. Attrition % Annualised
  3. Avg Salary Per FTE Annual (INR)
  4. Employee Engagement Cost (INR)    ← SARIMAX with closing_hc as exog
  5. Other Indirect Cost (INR)

Walk-forward validation (2 rounds) is also run to measure accuracy.
"""

from __future__ import annotations
import json
import pickle
import warnings
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── SARIMA order catalogue ────────────────────────────────────────────────
# (p,d,q)(P,D,Q,s)
SARIMA_ORDERS: dict[str, tuple] = {
    "closing_hc":       ((1, 1, 0), (1, 0, 0, 12)),
    "attrition_pct":    ((1, 1, 1), (1, 1, 0, 12)),
    "avg_salary":       ((0, 1, 1), (0, 1, 0, 12)),
    "engagement":       ((1, 1, 0), (1, 0, 0, 12)),   # uses HC as exog
    "other_indirect":   ((1, 1, 0), (0, 1, 0, 12)),
}

SERIES_MAP: dict[str, str] = {
    "closing_hc":     "Closing HC",
    "attrition_pct":  "Attrition % Annualized",
    "avg_salary":     "Avg Salary Per FTE Annual (INR)",
    "engagement":     "Employee Engagement Cost (INR)",
    "other_indirect": "Other Indirect Cost (INR)",
}

MODELS_DIR = Path(__file__).parent.parent / "models"

# Lightweight candidate catalogue around current defaults.
CANDIDATE_ORDERS: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]] = {
    "closing_hc": [
        ((1, 1, 0), (1, 0, 0, 12)),
        ((0, 1, 1), (1, 0, 0, 12)),
        ((1, 1, 1), (0, 1, 0, 12)),
        ((0, 1, 0), (1, 0, 0, 12)),
        ((0, 1, 0), (0, 0, 0, 0)),
    ],
    "attrition_pct": [
        ((1, 1, 1), (1, 1, 0, 12)),
        ((0, 1, 1), (1, 1, 0, 12)),
        ((1, 1, 0), (1, 1, 0, 12)),
        ((1, 0, 0), (0, 1, 0, 12)),
        ((0, 1, 0), (0, 0, 0, 0)),
    ],
    "avg_salary": [
        ((0, 1, 1), (0, 1, 0, 12)),
        ((1, 1, 0), (0, 1, 0, 12)),
        ((1, 1, 1), (0, 1, 0, 12)),
        ((0, 1, 1), (1, 0, 0, 12)),
        ((0, 1, 0), (0, 0, 0, 0)),
    ],
    "engagement": [
        ((1, 1, 0), (1, 0, 0, 12)),
        ((0, 1, 1), (1, 0, 0, 12)),
        ((1, 1, 1), (0, 1, 0, 12)),
        ((1, 0, 0), (1, 0, 0, 12)),
        ((0, 1, 0), (0, 0, 0, 0)),
    ],
    "other_indirect": [
        ((1, 1, 0), (0, 1, 0, 12)),
        ((0, 1, 1), (0, 1, 0, 12)),
        ((1, 1, 1), (0, 1, 0, 12)),
        ((1, 0, 0), (1, 0, 0, 12)),
        ((0, 1, 0), (0, 0, 0, 0)),
    ],
}


def _fit_sarima(
    series: pd.Series,
    order: tuple,
    seasonal_order: tuple,
    exog: pd.Series | None = None,
) -> Any:
    """Fit a SARIMAX model and return the fitted result handle."""
    try:
        mod = SARIMAX(
            series,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return mod.fit(disp=False, maxiter=200)
    except Exception as e:
        logger.warning(f"SARIMA fit failed: {e}. Falling back to (0,1,0)(0,0,0,0).")
        mod = SARIMAX(series, order=(0, 1, 0), seasonal_order=(0, 0, 0, 0),
                      enforce_stationarity=False, enforce_invertibility=False)
        return mod.fit(disp=False)


def _wmape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denom = float(np.sum(np.abs(actual)))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(actual - predicted)) / denom * 100)


def _seasonal_naive_from_train(
    train_series: pd.Series,
    test_index: pd.Index,
    season_len: int = 12,
) -> np.ndarray:
    if train_series.empty:
        return np.array([])
    hist = train_series.astype(float).copy()
    last_val = float(hist.iloc[-1])
    preds: list[float] = []
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


def _try_fit_candidate(
    series: pd.Series,
    order: tuple,
    seasonal_order: tuple,
    exog: pd.Series | None = None,
) -> Any | None:
    try:
        mod = SARIMAX(
            series,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = mod.fit(disp=False, maxiter=120)
        if not np.isfinite(getattr(fit, "aic", np.nan)):
            return None
        return fit
    except Exception:
        return None


def fit_best_sarima(
    series: pd.Series,
    key: str,
    exog: pd.Series | None = None,
) -> tuple[Any, tuple, tuple, float | None]:
    """
    Fit candidate SARIMA configurations and return:
      (best_fit, best_order, best_seasonal_order, best_aic)
    """
    candidates = CANDIDATE_ORDERS.get(key, [SARIMA_ORDERS[key]])
    best_fit = None
    best_order = SARIMA_ORDERS[key][0]
    best_seasonal = SARIMA_ORDERS[key][1]
    best_aic = np.inf

    for order, seasonal in candidates:
        fit = _try_fit_candidate(series, order, seasonal, exog=exog)
        if fit is None:
            continue
        if fit.aic < best_aic:
            best_aic = fit.aic
            best_fit = fit
            best_order = order
            best_seasonal = seasonal

    if best_fit is None:
        fallback_order, fallback_seasonal = SARIMA_ORDERS[key]
        best_fit = _fit_sarima(series, fallback_order, fallback_seasonal, exog=exog)
        best_order = fallback_order
        best_seasonal = fallback_seasonal
        best_aic = float(getattr(best_fit, "aic", np.nan))

    return best_fit, best_order, best_seasonal, (float(best_aic) if np.isfinite(best_aic) else None)


def learn_series_strategy(
    df: pd.DataFrame,
    min_lift_wmape: float = 1.0,
    holdout_months: int = 6,
    min_train_months: int = 18,
) -> dict:
    """
    Choose per-series strategy ('sarima' or 'naive') by comparing
    SARIMA vs seasonal-naive on a train-only holdout window.
    """
    strategies: dict = {}
    for key, col in SERIES_MAP.items():
        if col not in df.columns:
            continue

        series = df[col].dropna().astype(float)
        if len(series) < min_train_months:
            strategies[key] = {
                "strategy": "sarima",
                "reason": "insufficient_history",
                "n_obs": int(len(series)),
            }
            continue

        h = min(holdout_months, max(3, len(series) // 4))
        split = len(series) - h
        tr = series.iloc[:split]
        te = series.iloc[split:]
        if len(tr) < 12 or len(te) == 0:
            strategies[key] = {
                "strategy": "sarima",
                "reason": "insufficient_holdout",
                "n_obs": int(len(series)),
            }
            continue

        exog_tr = exog_te = None
        if key == "engagement" and "Closing HC" in df.columns:
            hc = df["Closing HC"].astype(float)
            exog_tr = hc.loc[tr.index].ffill()
            exog_te = hc.loc[te.index].ffill()

        try:
            fit, _, _, _ = fit_best_sarima(tr, key, exog=exog_tr)
            sar_pred = fit.forecast(steps=len(te), exog=exog_te if exog_te is not None else None).values[:len(te)]
            naive_pred = _seasonal_naive_from_train(tr, te.index)

            sar_wmape = _wmape(te.values, sar_pred)
            naive_wmape = _wmape(te.values, naive_pred)
            lift = naive_wmape - sar_wmape if np.isfinite(naive_wmape) and np.isfinite(sar_wmape) else np.nan
            chosen = "sarima" if (not np.isfinite(lift) or lift >= min_lift_wmape) else "naive"

            strategies[key] = {
                "strategy": chosen,
                "sarima_wmape": round(float(sar_wmape), 2) if np.isfinite(sar_wmape) else None,
                "naive_wmape": round(float(naive_wmape), 2) if np.isfinite(naive_wmape) else None,
                "lift_wmape_pts": round(float(lift), 2) if np.isfinite(lift) else None,
                "lift_threshold": float(min_lift_wmape),
                "holdout_months": int(len(te)),
                "n_obs": int(len(series)),
            }
        except Exception as e:
            strategies[key] = {
                "strategy": "sarima",
                "reason": f"selection_error: {e}",
                "n_obs": int(len(series)),
            }

    return strategies


def save_series_strategy(strategy: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(strategy, f, indent=2)


def train_department(df: pd.DataFrame, dept: str, models_dir: Path = MODELS_DIR) -> dict:
    """
    Train 5 SARIMA models for a department.
    Saves each as  models/sarima_{series}_{dept_slug}.pkl
    Returns a dict of fitted results for immediate use.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    slug = dept.lower().replace(" ", "_").replace("&", "and")
    fitted: dict = {}
    selected_orders: dict[str, dict] = {}

    for key, col in SERIES_MAP.items():
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found for {dept}, skipping.")
            continue

        series = df[col].dropna().astype(float)
        exog = None
        if key == "engagement" and "Closing HC" in df.columns:
            exog = df.loc[series.index, "Closing HC"].ffill().astype(float)

        logger.info(f"  Fitting {key} for {dept} (order search) …")
        result, order, seasonal, best_aic = fit_best_sarima(series, key, exog=exog)

        path = models_dir / f"sarima_{key}_{slug}.pkl"
        with open(path, "wb") as f:
            pickle.dump(result, f)

        fitted[key] = result
        selected_orders[key] = {
            "column": col,
            "order": list(order),
            "seasonal_order": list(seasonal),
            "aic": round(best_aic, 2) if best_aic is not None else None,
            "n_obs": int(len(series)),
        }

    if selected_orders:
        with open(models_dir / f"sarima_selected_orders_{slug}.json", "w") as f:
            json.dump(selected_orders, f, indent=2)

    return fitted


def load_department_models(dept: str, models_dir: Path = MODELS_DIR) -> dict:
    """Load all 5 pickled SARIMA models for a department."""
    slug = dept.lower().replace(" ", "_").replace("&", "and")
    loaded = {}
    for key in SARIMA_ORDERS:
        path = models_dir / f"sarima_{key}_{slug}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                loaded[key] = pickle.load(f)
        else:
            logger.warning(f"Model file not found: {path}")
    return loaded


# ── Walk-forward validation ───────────────────────────────────────────────

def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _direction_acc(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2:
        return np.nan
    a_diff = np.diff(actual)
    p_diff = np.diff(predicted)
    return float(np.mean(np.sign(a_diff) == np.sign(p_diff)) * 100)


def walk_forward_validate(df: pd.DataFrame, dept: str) -> dict:
    """
    2-round walk-forward:
      Round 1: train Apr-22→Sep-24, test Oct-24→Dec-24  (3 months)
      Round 2: train Apr-22→Dec-24, test Jan-25→Mar-25  (3 months)

    Returns metrics per series per round.
    """
    results: dict = {}
    slug = dept.lower().replace(" ", "_").replace("&", "and")

    splits = [
        ("2022-04", "2024-09", "2024-10", "2024-12"),
        ("2022-04", "2024-12", "2025-01", "2025-03"),
    ]

    for rnd, (train_start, train_end, test_start, test_end) in enumerate(splits, 1):
        round_key = f"round_{rnd}"
        results[round_key] = {}

        train = df.loc[train_start:train_end]
        test  = df.loc[test_start:test_end]
        if len(train) < 12 or len(test) == 0:
            continue
        h = len(test)

        for key, col in SERIES_MAP.items():
            if col not in df.columns:
                continue
            tr = train[col].dropna().astype(float)
            te = test[col].dropna().astype(float)
            if len(tr) < 12 or len(te) == 0:
                continue

            exog_tr = exog_te = None
            if key == "engagement" and "Closing HC" in df.columns:
                exog_tr = train.loc[tr.index, "Closing HC"].ffill().astype(float)
                exog_te = test.loc[te.index, "Closing HC"].ffill().astype(float)

            try:
                fit, best_order, best_seasonal, _ = fit_best_sarima(tr, key, exog=exog_tr)
                fc  = fit.forecast(steps=h, exog=exog_te if exog_te is not None else None)
                actual = te.values[:h]
                pred   = fc.values[:h]

                results[round_key][key] = {
                    "mape":      _mape(actual, pred),
                    "mae":       _mae(actual, pred),
                    "dir_acc":   _direction_acc(actual, pred),
                    "selected_order": list(best_order),
                    "selected_seasonal_order": list(best_seasonal),
                }
            except Exception as e:
                results[round_key][key] = {"error": str(e)}

    return results


def save_metrics(metrics: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
