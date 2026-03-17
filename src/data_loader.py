"""
data_loader.py
==============
Reads workbook data and returns a clean master DataFrame
with a (department, month_period) MultiIndex.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── constants ──────────────────────────────────────────────────────────────
DEPARTMENTS = [
    "Technology",
    "Operations",
    "Finance",
    "HR",
    "Sales & Marketing",
    "Risk & Compliance",
]

RAW_COLS = [
    "Month", "Year", "Opening HC", "Hires", "Exits", "Attrition",
    "Closing HC", "Long Leave Count", "Long Leave Cost (INR)",
    "Band A Count", "Band B Count", "Band C Count", "Band D Count", "Band E Count", "Band F Count",
    "Attrition % Annualized", "Increment % Applied", "Forex Rate USD/INR",
    "Avg Salary Per FTE Annual (INR)", "WFH Count", "WFO Count",
    "Direct Salary Cost (INR)", "Benefits Cost (INR)", "Variable Pay Bonus (INR)",
    "Payroll Tax (INR)", "Travel Allowance (INR)", "Meal Allowance (INR)",
    "Overtime Cost (INR)",
    "Recruitment Cost (INR)", "Training & Dev Cost (INR)", "IT License Cost (INR)",
    "IT Equipment Cost (INR)", "Rent & Facilities Cost (INR)", "Utilities Cost (INR)",
    "Admin Overhead (INR)", "HR Payroll Admin (INR)", "Learning Platform Cost (INR)",
    "Employee Engagement Cost (INR)", "Other Indirect Cost (INR)",
    "Total Direct Cost (INR)", "Total Indirect Cost (INR)", "Total Actual Cost (INR)",
    "Original Budget (INR)", "FY Annual Budget Pool (INR)",
    "Remaining Pool - Start of Month (INR)", "Months Remaining in FY",
    "Effective Monthly Budget (INR)", "Monthly Variance (INR)",
    "Remaining Pool - End of Month (INR)", "Cumulative Budget Used %",
    "Over / Under Flag",
]

DIRECT_COST_COLS = [
    "Direct Salary Cost (INR)", "Benefits Cost (INR)", "Variable Pay Bonus (INR)",
    "Payroll Tax (INR)", "Travel Allowance (INR)", "Meal Allowance (INR)",
    "Overtime Cost (INR)",
]

INDIRECT_COST_COLS = [
    "Recruitment Cost (INR)", "Training & Dev Cost (INR)", "IT License Cost (INR)",
    "IT Equipment Cost (INR)", "Rent & Facilities Cost (INR)", "Utilities Cost (INR)",
    "Admin Overhead (INR)", "HR Payroll Admin (INR)", "Learning Platform Cost (INR)",
    "Employee Engagement Cost (INR)", "Other Indirect Cost (INR)",
]


ALIAS_MAP: dict[str, list[str]] = {
    "Band A Count": ["Band A Count", "Band 1 Count"],
    "Band B Count": ["Band B Count", "Band 2 Count"],
    "Band C Count": ["Band C Count", "Band 3 Count"],
    "Band D Count": ["Band D Count", "Band 4 Count"],
    "Band E Count": ["Band E Count", "Band 5 Count"],
    "Band F Count": ["Band F Count", "Band 6 Count"],
    "Hires": ["Hires", "Total Hires"],
    "Exits": ["Exits", "Total Exits"],
    "Attrition": ["Attrition", "Total Exits"],
    "Long Leave Cost (INR)": ["Long Leave Cost (INR)"],
    "Attrition % Annualized": ["Attrition % Annualized", "Attrition % (Ann.)"],
    "Increment % Applied": ["Increment % Applied", "Avg Appraisal Rate"],
    "Forex Rate USD/INR": ["Forex Rate USD/INR", "CAD/INR Forex Rate"],
    "Avg Salary Per FTE Annual (INR)": ["Avg Salary Per FTE Annual (INR)", "Avg Salary (Wtd) (INR)"],
    "Direct Salary Cost (INR)": ["Direct Salary Cost (INR)", "Salary Cost (INR)"],
    "Variable Pay Bonus (INR)": ["Variable Pay Bonus (INR)", "Bonus/Var Pay (INR)"],
    "Rent & Facilities Cost (INR)": ["Rent & Facilities Cost (INR)", "Office Rent & Facilities (INR)"],
    "HR Payroll Admin (INR)": ["HR Payroll Admin (INR)", "HR Cost (INR)"],
    "Employee Engagement Cost (INR)": ["Employee Engagement Cost (INR)", "Emp Engagement Cost (INR)"],
    "Over / Under Flag": ["Over / Under Flag", "Over/Under"],
}

OTHER_INDIRECT_COMPONENTS = [
    "DBTS Charges (INR)",
    "Consultancy Charges (INR)",
    "Financial Operation Charges (INR)",
    "Mgmt Overhead (INR)",
]


def _coalesce(raw: pd.DataFrame, candidates: list[str], default=np.nan) -> pd.Series:
    for c in candidates:
        if c in raw.columns:
            return raw[c]
    return pd.Series([default] * len(raw))


def _build_period(month_series: pd.Series, year_series: pd.Series) -> pd.PeriodIndex:
    month_str = month_series.astype(str).str.strip()
    year_num = pd.to_numeric(year_series, errors="coerce")

    # Works for "Jan", "Apr", etc.
    month_num = pd.to_datetime(month_str.str[:3], format="%b", errors="coerce").dt.month

    # Fallback for month strings that already contain year, e.g. "Apr-2022".
    needs_fallback = month_str.str.contains(r"\d{4}", regex=True, na=False)
    month_period = pd.Series(pd.NaT, index=month_str.index, dtype="datetime64[ns]")
    if needs_fallback.any():
        month_period.loc[needs_fallback] = pd.to_datetime(
            month_str.loc[needs_fallback],
            errors="coerce",
        )
    year_fallback = pd.Series(month_period.dt.year, index=month_str.index)
    month_num_fallback = pd.Series(month_period.dt.month, index=month_str.index)

    year_final = year_num.fillna(year_fallback)
    month_final = month_num.fillna(month_num_fallback)

    dt = pd.to_datetime(
        {"year": year_final, "month": month_final, "day": 1},
        errors="coerce",
    )
    return dt.dt.to_period("M")


def load_department(xl: pd.ExcelFile, dept: str) -> pd.DataFrame:
    """Parse one department sheet into a clean DataFrame."""
    raw = xl.parse(dept, header=None)

    # Row 0 = dept title (metadata), Row 1 = actual column headers
    raw.columns = raw.iloc[1]
    raw = raw.iloc[2:].reset_index(drop=True)

    # Normalize headers and build canonical dataframe with schema aliases.
    raw.columns = [str(c).strip() for c in raw.columns]
    df = pd.DataFrame(index=raw.index)
    for canonical in RAW_COLS:
        if canonical in raw.columns:
            df[canonical] = raw[canonical]
        elif canonical in ALIAS_MAP:
            df[canonical] = _coalesce(raw, ALIAS_MAP[canonical])
        else:
            df[canonical] = np.nan

    # If explicit Other Indirect doesn't exist in source, aggregate residual buckets.
    if "Other Indirect Cost (INR)" in df.columns:
        if pd.to_numeric(df["Other Indirect Cost (INR)"], errors="coerce").isna().all():
            present = [c for c in OTHER_INDIRECT_COMPONENTS if c in raw.columns]
            if present:
                comp = raw[present].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                df["Other Indirect Cost (INR)"] = comp.sum(axis=1)

    # Drop completely empty rows
    df = df.dropna(subset=["Month"]).reset_index(drop=True)

    # Coerce numerics
    num_cols = [c for c in RAW_COLS if c not in ("Month", "Year", "Over / Under Flag")]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # Period index
    df["Period"] = _build_period(df["Month"], df["Year"])
    df = df.dropna(subset=["Period"]).reset_index(drop=True)
    df["Dept"] = dept
    df = df.set_index("Period").sort_index()

    # Derived helpers used frequently
    # Jan-Dec fiscal year: fiscal month aligns with calendar month.
    df["month_in_fy"] = df.index.month
    df["fy_label"] = df.index.year

    return df


def load_all(xlsx_path: str | Path) -> dict[str, pd.DataFrame]:
    """
    Returns a dict  {dept_name: DataFrame}  for all 6 departments.
    Each DataFrame is indexed by pd.Period (monthly).
    """
    xl = pd.ExcelFile(xlsx_path)
    return {dept: load_department(xl, dept) for dept in DEPARTMENTS}


def get_annual_budgets(dept_df: pd.DataFrame) -> pd.Series:
    """
    Extract unique FY annual budget pool per FY year from a dept DataFrame.
    Uses the first record of each January (FY start) → 'FY Annual Budget Pool (INR)'.
    Returns a Series indexed by FY year (e.g. 2024, 2025, 2026).
    """
    jan = dept_df[dept_df["month_in_fy"] == 1]
    budget = jan["FY Annual Budget Pool (INR)"].dropna()
    return budget


if __name__ == "__main__":
    path = Path(__file__).parent.parent / "data" / "raw" / "workforce_cost_model_v4.xlsx"
    depts = load_all(path)
    for name, df in depts.items():
        print(f"{name}: {df.shape}  index {df.index[0]}…{df.index[-1]}")
