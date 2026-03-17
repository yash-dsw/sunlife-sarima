"""
cost_rules.py
=============
Learns per-FTE / per-hire / per-event rates from historical data,
then applies them as a pure formula engine to produce predicted costs.

The only SARIMA-predicted values fed in from outside are:
  - closing_hc
  - attrition_pct_annualised
  - avg_salary_per_fte_annual
  - employee_engagement_cost
  - other_indirect_cost

Everything else is a formula.
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path


# ── constants ─────────────────────────────────────────────────────────────
STATUTORY_BENEFITS_RATE   = 0.18
VARIABLE_PAY_RATE         = 0.08
PAYROLL_TAX_RATE          = 0.135
MEAL_PER_HC               = 1_200      # INR per head per month
TRAVEL_PER_WFO            = 1_500      # INR per WFO-head per month
DEFAULT_WFH_PCT           = 0.55       # fallback if not available


# ── rate learner ──────────────────────────────────────────────────────────

def learn_rates(df: pd.DataFrame, lookback_months: int = 18) -> dict:
    """
    Derive per-unit cost rates by regressing historical data.
    Returns a dict of learned scalars that the rules engine uses.
    """
    if lookback_months and len(df) > lookback_months:
        df = df.tail(lookback_months).copy()

    rates: dict = {}

    def _safe_mean_ratio(num_col: str, denom_col: str) -> float:
        mask = df[denom_col] > 0
        ratios = df.loc[mask, num_col] / df.loc[mask, denom_col]
        return float(ratios.replace([np.inf, -np.inf], np.nan).dropna().mean())

    # ── payroll composition rates (dataset-specific) ────────────────────
    rates["benefits_rate"] = _safe_mean_ratio("Benefits Cost (INR)", "Direct Salary Cost (INR)")
    rates["variable_pay_rate"] = _safe_mean_ratio("Variable Pay Bonus (INR)", "Direct Salary Cost (INR)")
    rates["payroll_tax_rate"] = _safe_mean_ratio("Payroll Tax (INR)", "Direct Salary Cost (INR)")
    rates["meal_per_hc"] = _safe_mean_ratio("Meal Allowance (INR)", "Closing HC")
    rates["travel_per_wfo"] = _safe_mean_ratio("Travel Allowance (INR)", "WFO Count")

    # ── headcount-driven ────────────────────────────────────────────────
    # Calibrate direct-salary baseline because Avg Salary may exclude some paid components
    # that are present in the dataset's Direct Salary Cost.
    salary_base = (df["Closing HC"] * df["Avg Salary Per FTE Annual (INR)"] / 12.0).replace([np.inf, -np.inf], np.nan)
    salary_mask = salary_base > 0
    salary_ratio = (df.loc[salary_mask, "Direct Salary Cost (INR)"] / salary_base.loc[salary_mask]).replace([np.inf, -np.inf], np.nan)
    salary_uplift = float(salary_ratio.dropna().mean()) if not salary_ratio.dropna().empty else 1.0
    rates["salary_uplift"] = float(np.clip(salary_uplift, 0.7, 1.6))

    rates["per_fte_it_license"]   = _safe_mean_ratio("IT License Cost (INR)", "Closing HC")
    rates["per_fte_admin"]        = _safe_mean_ratio("Admin Overhead (INR)", "Closing HC")
    rates["per_fte_hr"]           = _safe_mean_ratio("HR Payroll Admin (INR)", "Closing HC")
    rates["per_fte_learning"]     = _safe_mean_ratio("Learning Platform Cost (INR)", "Closing HC")

    # WFO-driven
    rates["per_wfo_rent"]         = _safe_mean_ratio("Rent & Facilities Cost (INR)", "WFO Count")
    rates["per_wfo_utilities"]    = _safe_mean_ratio("Utilities Cost (INR)", "WFO Count")

    # attrition / hire-driven
    rates["cost_per_hire"]        = _safe_mean_ratio("Recruitment Cost (INR)", "Hires")
    rates["cost_per_device"]      = _safe_mean_ratio("IT Equipment Cost (INR)", "Hires")
    rates["overtime_per_exit"]    = _safe_mean_ratio("Overtime Cost (INR)", "Exits")

    # Training: split into onboarding part (per hire) + base (per HC)
    # Training = hires × onboard_rate + closing_hc × base_rate
    # Solve via OLS on two variables
    X = df[["Hires", "Closing HC"]].fillna(0).values
    y = df["Training & Dev Cost (INR)"].fillna(0).values
    if X.shape[0] > 2:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        rates["training_per_hire"]    = max(0.0, float(coef[0]))
        rates["training_per_hc_base"] = max(0.0, float(coef[1]))
    else:
        rates["training_per_hire"]    = 50_000.0
        rates["training_per_hc_base"] = 2_000.0

    # WFH % (historical average)
    if "WFH Count" in df.columns and "Closing HC" in df.columns:
        mask = df["Closing HC"] > 0
        wfh_series = df.loc[mask, "WFH Count"] / df.loc[mask, "Closing HC"]
        rates["wfh_pct"] = float(wfh_series.dropna().mean()) if not wfh_series.empty else DEFAULT_WFH_PCT
    else:
        rates["wfh_pct"] = DEFAULT_WFH_PCT

    # Seasonal multipliers for engagement (12-element list, index 0 = Jan)
    if "Employee Engagement Cost (INR)" in df.columns and "Closing HC" in df.columns:
        df2 = df.copy()
        df2["eng_per_fte"] = np.where(
            df2["Closing HC"] > 0,
            df2["Employee Engagement Cost (INR)"] / df2["Closing HC"],
            np.nan,
        )
        monthly_avg = df2.groupby(df2.index.month)["eng_per_fte"].mean()
        all_months  = {m: monthly_avg.get(m, monthly_avg.mean()) for m in range(1, 13)}
        rates["engagement_seasonal_per_fte"] = {str(k): v for k, v in all_months.items()}
    else:
        rates["engagement_seasonal_per_fte"] = {str(m): 600.0 for m in range(1, 13)}

    rates["rates_lookback_months"] = int(min(len(df), lookback_months)) if lookback_months else int(len(df))
    return rates


# ── seasonal engagement helper ─────────────────────────────────────────────

def _engagement_from_rates(closing_hc: float, month: int, rates: dict) -> float:
    key = str(month)
    per_fte = rates["engagement_seasonal_per_fte"].get(key, 600.0)
    return closing_hc * per_fte


# ── main rules engine ─────────────────────────────────────────────────────

def compute_costs(
    closing_hc:          float,
    opening_hc:          float,
    attrition_pct:       float,   # annualised decimal e.g. 0.15
    avg_salary_annual:   float,   # INR per FTE per year
    month:               int,     # calendar month (1-12)
    rates:               dict,
    wfh_pct:             float | None = None,
    # optional SARIMA-predicted overrides
    engagement_override: float | None = None,
    other_indirect_override: float | None = None,
) -> dict:
    """
    Pure formula engine.  Returns a flat dict of all cost items + aggregates.
    """
    wfh_pct  = wfh_pct if wfh_pct is not None else rates.get("wfh_pct", DEFAULT_WFH_PCT)
    wfo_count = max(0.0, closing_hc * (1 - wfh_pct))

    # ── HC movements ────────────────────────────────────────────────────
    exits = round(opening_hc * attrition_pct / 12)
    hires = max(0.0, closing_hc - opening_hc + exits)

    # ── direct salary base ────────────────────────────────────────────
    salary_uplift = float(rates.get("salary_uplift", 1.0))
    monthly_salary_base = closing_hc * avg_salary_annual / 12 * salary_uplift

    direct_salary     = monthly_salary_base
    benefits_rate     = rates.get("benefits_rate", STATUTORY_BENEFITS_RATE)
    variable_rate     = rates.get("variable_pay_rate", VARIABLE_PAY_RATE)
    payroll_rate      = rates.get("payroll_tax_rate", PAYROLL_TAX_RATE)
    meal_per_hc       = rates.get("meal_per_hc", MEAL_PER_HC)
    travel_per_wfo    = rates.get("travel_per_wfo", TRAVEL_PER_WFO)

    benefits          = direct_salary * benefits_rate
    variable_pay      = direct_salary * variable_rate
    payroll_tax       = direct_salary * payroll_rate
    travel_allowance  = wfo_count * travel_per_wfo
    meal_allowance    = closing_hc * meal_per_hc
    overtime_rate     = rates.get("overtime_per_exit", 30_000.0)
    overtime          = exits * overtime_rate

    total_direct = (
        direct_salary + benefits + variable_pay + payroll_tax
        + travel_allowance + meal_allowance + overtime
    )

    # ── indirect ──────────────────────────────────────────────────────
    recruitment    = hires * rates.get("cost_per_hire", 35_000.0)
    training_dev   = (
        hires * rates.get("training_per_hire", 50_000.0)
        + closing_hc * rates.get("training_per_hc_base", 2_000.0)
    )
    it_license     = closing_hc * rates.get("per_fte_it_license", 3_000.0)
    it_equipment   = hires * rates.get("cost_per_device", 45_000.0)
    rent           = wfo_count * rates.get("per_wfo_rent", 7_000.0)
    utilities      = wfo_count * rates.get("per_wfo_utilities", 1_200.0)
    admin_overhead = closing_hc * rates.get("per_fte_admin", 1_500.0)
    hr_admin       = closing_hc * rates.get("per_fte_hr", 800.0)
    learning       = closing_hc * rates.get("per_fte_learning", 600.0)

    # engagement: SARIMA or seasonal rule
    engagement = (
        engagement_override
        if engagement_override is not None
        else _engagement_from_rates(closing_hc, month, rates)
    )

    other_indirect = (
        other_indirect_override
        if other_indirect_override is not None
        else 0.0
    )

    total_indirect = (
        recruitment + training_dev + it_license + it_equipment
        + rent + utilities + admin_overhead + hr_admin + learning
        + engagement + other_indirect
    )
    total_cost = total_direct + total_indirect

    return dict(
        closing_hc=closing_hc,
        opening_hc=opening_hc,
        exits=exits,
        hires=hires,
        wfo_count=wfo_count,
        wfh_pct=wfh_pct,
        # direct
        direct_salary_cost=direct_salary,
        benefits_cost=benefits,
        variable_pay_bonus=variable_pay,
        payroll_tax=payroll_tax,
        travel_allowance=travel_allowance,
        meal_allowance=meal_allowance,
        overtime_cost=overtime,
        total_direct_cost=total_direct,
        # indirect
        recruitment_cost=recruitment,
        training_dev_cost=training_dev,
        it_license_cost=it_license,
        it_equipment_cost=it_equipment,
        rent_facilities_cost=rent,
        utilities_cost=utilities,
        admin_overhead=admin_overhead,
        hr_payroll_admin=hr_admin,
        learning_platform_cost=learning,
        employee_engagement_cost=engagement,
        other_indirect_cost=other_indirect,
        total_indirect_cost=total_indirect,
        total_cost=total_cost,
    )


# ── persist / load ────────────────────────────────────────────────────────

def save_rates(rates: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rates, f, indent=2)


def load_rates(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
