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
    """
    pool = float(start_pool if start_pool is not None else annual_budget)
    n    = len(monthly_costs)
    rows = []

    for i, period in enumerate(periods):
        months_left        = n - i
        eff_monthly_budget = pool / months_left if months_left > 0 else 0.0
        cost               = float(monthly_costs[i])
        variance           = eff_monthly_budget - cost
        pool_end           = pool - cost
        flag               = "✓ Within Budget" if cost <= eff_monthly_budget else "⚠ Over Budget"

        rows.append({
            "Period":                   str(period),
            "Effective Monthly Budget":  round(eff_monthly_budget, 2),
            "Predicted Cost":            round(cost, 2),
            "Pool (Start of Month)":     round(pool, 2),
            "Pool (End of Month)":       round(pool_end, 2),
            "Monthly Variance":          round(variance, 2),
            "Flag":                      flag,
        })
        pool = pool_end

    df = pd.DataFrame(rows)
    df["Period"] = pd.PeriodIndex(df["Period"], freq="M")
    df = df.set_index("Period")
    return df
