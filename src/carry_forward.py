"""
carry_forward.py
================
Budget pool carry-forward (rephasing) engine.

Given:
  annual_budget   : total FY budget for the department
  monthly_costs   : list/array of predicted monthly costs (length 12)
  start_pool      : optional  — defaults to annual_budget
  start_month_idx : 0-based index of the first month (0=Apr if Apr FY)

Returns a DataFrame with columns:
  Month | Monthly Budget (Effective) | Predicted Cost | Pool Remaining | Over/Under
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def run_carry_forward(
    annual_budget:   float,
    monthly_costs:   list | np.ndarray,
    periods:         pd.PeriodIndex,
    start_pool:      float | None = None,
    planned_monthly_budget: list | np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Rephasing rule:
      effective_monthly_budget = remaining_pool / months_left

    Parameters
    ----------
    annual_budget  : total FY pool
    monthly_costs  : predicted cost per month (length == len(periods))
    periods        : PeriodIndex of the forecast months
    start_pool     : override starting pool (useful in mid-year runs)
    planned_monthly_budget : optional monthly budget forecast. If provided,
                            carry-forward keeps this monthly shape and rescales
                            by remaining pool.
    """
    pool = float(start_pool if start_pool is not None else annual_budget)
    n    = len(monthly_costs)
    planned = None
    if planned_monthly_budget is not None:
        arr = np.asarray(planned_monthly_budget, dtype=float)
        if len(arr) == n:
            planned = np.maximum(0.0, arr)
    rows = []

    for i, period in enumerate(periods):
        months_left        = n - i
        if planned is not None:
            planned_budget_i = float(planned[i])
            remaining_plan_sum = float(np.sum(planned[i:]))
            if remaining_plan_sum > 0:
                eff_monthly_budget = pool * (float(planned[i]) / remaining_plan_sum)
            else:
                eff_monthly_budget = pool / months_left if months_left > 0 else 0.0
        else:
            planned_budget_i = np.nan
            eff_monthly_budget = pool / months_left if months_left > 0 else 0.0
        cost               = float(monthly_costs[i])
        effective_variance = eff_monthly_budget - cost
        planned_variance   = (planned_budget_i - cost) if not np.isnan(planned_budget_i) else np.nan
        pool_end           = pool - cost
        effective_flag     = "✓ Within Budget" if cost <= eff_monthly_budget else "⚠ Over Budget"
        if not np.isnan(planned_budget_i):
            flag = "✓ Within Budget" if cost <= planned_budget_i else "⚠ Over Budget"
        else:
            flag = effective_flag

        row = {
            "Period":                   str(period),
            "Effective Monthly Budget":  round(eff_monthly_budget, 2),
            "Predicted Cost":            round(cost, 2),
            "Pool (Start of Month)":     round(pool, 2),
            "Pool (End of Month)":       round(pool_end, 2),
            "Monthly Variance":          round(planned_variance if not np.isnan(planned_variance) else effective_variance, 2),
            "Flag":                      flag,
            "Effective Variance":        round(effective_variance, 2),
            "Effective Flag":            effective_flag,
        }
        if planned is not None:
            row["Planned Monthly Budget"] = round(planned_budget_i, 2)
            row["Planned Variance"] = round(planned_variance, 2)
        rows.append(row)
        pool = pool_end

    df = pd.DataFrame(rows)
    df["Period"] = pd.PeriodIndex(df["Period"], freq="M")
    df = df.set_index("Period")
    return df
