"""
data_loader.py
==============
Reads workforce_cost_model_v4.xlsx and returns a clean master DataFrame
with a (department, month_period) MultiIndex.
"""

import pandas as pd
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
    "Band A Count", "Band B Count", "Band C Count", "Band D Count", "Band E Count",
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


def _parse_month(s: str) -> pd.Period:
    """Convert 'Apr-2022' → Period('2022-04', 'M')."""
    return pd.Period(s, freq="M")


def load_department(xl: pd.ExcelFile, dept: str) -> pd.DataFrame:
    """Parse one department sheet into a clean DataFrame."""
    raw = xl.parse(dept, header=None)

    # Row 0 = dept title (metadata), Row 1 = actual column headers
    raw.columns = raw.iloc[1]
    raw = raw.iloc[2:].reset_index(drop=True)

    # Keep only recognised columns
    available = [c for c in RAW_COLS if c in raw.columns]
    df = raw[available].copy()

    # Drop completely empty rows
    df = df.dropna(subset=["Month"]).reset_index(drop=True)

    # Coerce numerics
    num_cols = [c for c in available if c not in ("Month", "Year", "Over / Under Flag")]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # Period index
    df["Period"] = df["Month"].apply(_parse_month)
    df["Dept"] = dept
    df = df.set_index("Period").sort_index()

    # Derived helpers used frequently
    df["month_in_fy"] = df.index.month          # Apr=4 … Mar=3
    df["fy_label"] = df.index.year              # calendar year of that period

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
    Uses the first record of each April (FY start) → 'FY Annual Budget Pool (INR)'.
    Returns a Series indexed by FY year (e.g. 2022, 2023, 2024).
    """
    apr = dept_df[dept_df["month_in_fy"] == 4]
    budget = apr["FY Annual Budget Pool (INR)"].dropna()
    return budget


if __name__ == "__main__":
    path = Path(__file__).parent.parent / "data" / "raw" / "workforce_cost_model_v4.xlsx"
    depts = load_all(path)
    for name, df in depts.items():
        print(f"{name}: {df.shape}  index {df.index[0]}…{df.index[-1]}")
