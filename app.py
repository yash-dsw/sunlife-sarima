"""
app.py
======
Streamlit UI for:
  1) Forecast simulation
  2) Model performance dashboard

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from src.carry_forward import run_carry_forward
from src.cost_rules import load_rates
from src.data_loader import DEPARTMENTS, load_all
from src.forecaster import extrapolate_annual_budget, forecast_dept
from src.model_trainer import SARIMA_ORDERS


BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "raw" / "workforce_cost_model_v4.xlsx"
MODELS_DIR = BASE_DIR / "models"
VALIDATION_PATH = BASE_DIR / "validation_report.csv"
TRAINING_METRICS_PATH = MODELS_DIR / "training_metrics.json"

DEPT_SLUG_MAP = {
    d.lower().replace(" ", "_").replace("&", "and"): d for d in DEPARTMENTS
}


@st.cache_resource(show_spinner="Loading historical data...")
def get_data():
    return load_all(DATA_PATH)


@st.cache_resource(show_spinner="Loading models...")
def get_models(dept: str):
    slug = dept.lower().replace(" ", "_").replace("&", "and")
    loaded = {}
    for key in SARIMA_ORDERS:
        path = MODELS_DIR / f"sarima_{key}_{slug}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                loaded[key] = pickle.load(f)
    return loaded


@st.cache_resource(show_spinner="Loading rates...")
def get_rates(dept: str):
    slug = dept.lower().replace(" ", "_").replace("&", "and")
    path = MODELS_DIR / f"per_fte_rates_{slug}.json"
    if path.exists():
        return load_rates(path)
    return {}


@st.cache_resource(show_spinner="Loading forecast strategy...")
def get_strategy(dept: str):
    slug = dept.lower().replace(" ", "_").replace("&", "and")
    path = MODELS_DIR / f"series_strategy_{slug}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


@st.cache_data(show_spinner=False)
def load_validation_report() -> pd.DataFrame:
    if not VALIDATION_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(VALIDATION_PATH)
    if "Evaluation Type" not in df.columns:
        df["Evaluation Type"] = "Rules-only (actual drivers)"
    return df


@st.cache_data(show_spinner=False)
def load_training_metrics_long() -> pd.DataFrame:
    if not TRAINING_METRICS_PATH.exists():
        return pd.DataFrame()
    raw = json.loads(TRAINING_METRICS_PATH.read_text())
    rows: list[dict] = []
    for dept, rounds in raw.items():
        for rnd, series_map in rounds.items():
            for series, metrics in series_map.items():
                if "error" in metrics:
                    continue
                rows.append(
                    {
                        "Dept": dept,
                        "Round": rnd,
                        "Series": series,
                        "MAPE %": metrics.get("mape"),
                        "MAE": metrics.get("mae"),
                        "DirAcc %": metrics.get("dir_acc"),
                    }
                )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_selected_orders_long() -> pd.DataFrame:
    rows: list[dict] = []
    for path in MODELS_DIR.glob("sarima_selected_orders_*.json"):
        dept_slug = path.stem.replace("sarima_selected_orders_", "")
        dept = DEPT_SLUG_MAP.get(dept_slug, dept_slug)
        try:
            raw = json.loads(path.read_text())
        except Exception:
            continue
        for series, meta in raw.items():
            rows.append(
                {
                    "Dept": dept,
                    "Series": series,
                    "Order": str(tuple(meta.get("order", []))),
                    "Seasonal": str(tuple(meta.get("seasonal_order", []))),
                    "AIC": meta.get("aic"),
                    "N Obs": meta.get("n_obs"),
                }
            )
    return pd.DataFrame(rows)


def _models_exist() -> bool:
    return any(MODELS_DIR.glob("sarima_*.pkl"))


def _fmt_inr(val: float) -> str:
    sign = "-" if val < 0 else ""
    abs_val = abs(float(val))
    if abs_val >= 1e7:
        return f"{sign}Rs {abs_val/1e7:.2f} Cr"
    if abs_val >= 1e5:
        return f"{sign}Rs {abs_val/1e5:.2f} L"
    return f"{sign}Rs {abs_val:,.0f}"


def _accuracy_band(mape_value: float) -> str:
    if pd.isna(mape_value):
        return "Unknown"
    if mape_value <= 10:
        return "Excellent"
    if mape_value <= 20:
        return "Good"
    if mape_value <= 30:
        return "Fair"
    return "Weak"


def _trend_band(dir_acc_value: float) -> str:
    if pd.isna(dir_acc_value):
        return "Unknown"
    if dir_acc_value >= 75:
        return "Strong"
    if dir_acc_value >= 60:
        return "Moderate"
    return "Weak"


def _render_performance_dashboard() -> None:
    st.title("Workforce Cost Model Performance")
    st.caption("Backtesting summary for trend detection and value prediction quality.")

    val_df = load_validation_report()
    series_df = load_training_metrics_long()
    orders_df = load_selected_orders_long()

    if val_df.empty:
        st.warning(
            "validation_report.csv was not found. Run `python evaluate.py --rounds 1 2 full --output validation_report.csv` first."
        )
    else:
        c1, c2, c3 = st.columns(3)
        rounds = sorted(val_df["Round"].dropna().unique().tolist())
        default_round = "Full Hold-out (Oct-24→Mar-25)" if "Full Hold-out (Oct-24→Mar-25)" in rounds else rounds[0]
        with c1:
            selected_round = st.selectbox("Evaluation window", rounds, index=rounds.index(default_round))
        eval_types = sorted(val_df["Evaluation Type"].dropna().unique().tolist())
        with c2:
            selected_eval = st.selectbox("Evaluation type", eval_types)
        cost_lines = sorted(val_df["Cost Line"].dropna().unique().tolist())
        default_cost = "Total Actual Cost (INR)" if "Total Actual Cost (INR)" in cost_lines else cost_lines[0]
        with c3:
            selected_cost = st.selectbox("Metric focus", cost_lines, index=cost_lines.index(default_cost))

        subset = val_df[
            (val_df["Round"] == selected_round)
            & (val_df["Evaluation Type"] == selected_eval)
            & (val_df["Cost Line"] == selected_cost)
        ].copy()

        if subset.empty:
            st.warning("No rows match this filter.")
        else:
            metric_cols = [
                "MAPE %",
                "WMAPE %",
                "MAE",
                "RMSE",
                "DirAcc %",
                "MaxErr",
                "MAPE Lift vs Naive %pts",
                "WMAPE Lift vs Naive %pts",
            ]
            for col in metric_cols:
                if col not in subset.columns:
                    subset[col] = pd.NA
                subset[col] = pd.to_numeric(subset[col], errors="coerce")

            avg_mape = float(subset["MAPE %"].mean())
            avg_wmape = float(subset["WMAPE %"].mean()) if "WMAPE %" in subset.columns else float("nan")
            avg_dir = float(subset["DirAcc %"].mean())
            avg_lift = float(subset["WMAPE Lift vs Naive %pts"].mean()) if "WMAPE Lift vs Naive %pts" in subset.columns else float("nan")
            best_row = subset.sort_values("MAPE %", ascending=True).iloc[0]
            worst_row = subset.sort_values("MAPE %", ascending=False).iloc[0]

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Average MAPE", f"{avg_mape:.2f}%")
            k2.metric("Average WMAPE", f"{avg_wmape:.2f}%")
            k3.metric("Best Department", f"{best_row['Dept']} ({best_row['MAPE %']:.2f}%)")
            k4.metric("WMAPE Lift vs Naive", f"{avg_lift:.2f} pts")
            st.caption(f"Average trend accuracy: {avg_dir:.1f}% | Weakest department: {worst_row['Dept']} ({worst_row['MAPE %']:.2f}%)")

            st.info(
                "Interpretation: MAPE<=10 Excellent, 10-20 Good, 20-30 Fair, >30 Weak. "
                "Use WMAPE for business-weighted error. Positive Lift vs Naive means your model beats seasonal naive."
            )

            view_df = subset[
                ["Dept"] + metric_cols
            ].copy()
            view_df["Accuracy Band"] = view_df["MAPE %"].apply(_accuracy_band)
            view_df["Trend Band"] = view_df["DirAcc %"].apply(_trend_band)
            view_df["MAE"] = view_df["MAE"].apply(_fmt_inr)
            view_df["RMSE"] = view_df["RMSE"].apply(_fmt_inr)
            view_df["MaxErr"] = view_df["MaxErr"].apply(_fmt_inr)
            st.dataframe(view_df, hide_index=True, use_container_width=True)

            c_mape, c_dir = st.columns(2)
            with c_mape:
                st.subheader("WMAPE by Department")
                wmape_chart = subset[["Dept", "WMAPE %"]].set_index("Dept")
                st.bar_chart(wmape_chart, use_container_width=True)
            with c_dir:
                st.subheader("WMAPE Lift vs Naive")
                lift_chart = subset[["Dept", "WMAPE Lift vs Naive %pts"]].set_index("Dept")
                st.bar_chart(lift_chart, use_container_width=True)

    st.subheader("Driver-Series Accuracy")
    if series_df.empty:
        st.warning("models/training_metrics.json not found. Run `python train.py` first.")
    else:
        round_opts = ["all"] + sorted(series_df["Round"].dropna().unique().tolist())
        selected_series_round = st.selectbox("Training round", round_opts, index=0)
        filtered = series_df.copy() if selected_series_round == "all" else series_df[series_df["Round"] == selected_series_round]

        agg = (
            filtered.groupby("Series", as_index=False)[["MAPE %", "DirAcc %"]]
            .mean()
            .sort_values("MAPE %", ascending=True)
        )
        agg["Accuracy Band"] = agg["MAPE %"].apply(_accuracy_band)
        agg["Trend Band"] = agg["DirAcc %"].apply(_trend_band)
        st.dataframe(agg, hide_index=True, use_container_width=True)

    st.subheader("Selected SARIMA Orders")
    if orders_df.empty:
        st.caption("Run `python train.py` to generate selected-order files.")
    else:
        dept_opts = sorted(orders_df["Dept"].unique().tolist())
        dept_choice = st.selectbox("Department (order details)", dept_opts, key="order_dept")
        od = orders_df[orders_df["Dept"] == dept_choice].copy().sort_values("Series")
        st.dataframe(od, hide_index=True, use_container_width=True)

    st.subheader("Validation Design Check")
    has_full_pipeline = (not val_df.empty) and (val_df["Evaluation Type"] == "Full pipeline (forecast drivers)").any()
    audit = pd.DataFrame(
        [
            {
                "Check": "Chronological split (train before test)",
                "Status": "OK",
                "Why it matters": "Prevents future leakage into training.",
            },
            {
                "Check": "Walk-forward rounds (Oct-Dec 2024, Jan-Mar 2025)",
                "Status": "OK",
                "Why it matters": "Tests model stability across two future windows.",
            },
            {
                "Check": "Rules-only evaluation (actual drivers)",
                "Status": "OK",
                "Why it matters": "Isolates rules-engine quality from SARIMA driver errors.",
            },
            {
                "Check": "Full pipeline evaluation (forecast drivers)",
                "Status": "OK" if has_full_pipeline else "Needs refresh",
                "Why it matters": "This is the true forecasting score for future prediction quality.",
            },
            {
                "Check": "Independent final test set beyond Mar-2025",
                "Status": "Missing",
                "Why it matters": "Current setup is backtesting only; keep one untouched period for final sign-off.",
            },
        ]
    )
    st.dataframe(audit, hide_index=True, use_container_width=True)


def _render_simulator(depts_data: dict[str, pd.DataFrame]) -> None:
    st.title("Workforce Cost Forecasting and Simulation")

    with st.sidebar:
        st.header("Department")
        dept = st.selectbox("Select Department", DEPARTMENTS, key="sim_dept")

        st.divider()
        st.subheader("Headcount and Attrition")

        dept_df = depts_data[dept]
        models = get_models(dept)
        rates = get_rates(dept)
        strategy = get_strategy(dept)

        last_hc = int(dept_df["Closing HC"].iloc[-1])
        last_att = float(dept_df["Attrition % Annualized"].iloc[-1])
        last_wfh = float(rates.get("wfh_pct", 0.55))
        suggested_budget = extrapolate_annual_budget(dept_df)

        defaults = {
            "sim_target_hc": last_hc,
            "sim_attrition_pct_ui": int(round(last_att * 100)),
            "sim_wfh_pct_ui": int(round(last_wfh * 100)),
            "sim_increment_pct_ui": 10,
            "sim_salary_override_raw": 0,
            "sim_annual_budget": int(round(suggested_budget)),
        }
        base_override = {
            "closing_hc": defaults["sim_target_hc"],
            "attrition_pct": defaults["sim_attrition_pct_ui"] / 100.0,
            "wfh_pct": defaults["sim_wfh_pct_ui"] / 100.0,
            "increment_pct": defaults["sim_increment_pct_ui"] / 100.0,
        }
        if defaults["sim_salary_override_raw"] > 0:
            base_override["avg_salary"] = float(defaults["sim_salary_override_raw"])

        # Apply pending reset before widgets are created.
        if st.session_state.pop("sim_reset_requested", False):
            for k, v in defaults.items():
                st.session_state[k] = v

        # Dept switch should load that dept's defaults, not carry old slider values.
        if st.session_state.get("sim_prev_dept") != dept:
            for k, v in defaults.items():
                st.session_state[k] = v
            st.session_state["sim_prev_dept"] = dept

        # Ensure keys exist on first run.
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

        st.number_input("Opening HC (locked)", value=last_hc, disabled=True)
        target_hc = st.slider(
            "Target Closing HC",
            min_value=1,
            max_value=last_hc * 3,
            help="Attrition reduces HC unless target closing HC is raised.",
            key="sim_target_hc",
        )
        attrition_pct_ui = st.slider(
            "Attrition % (annualised)",
            min_value=0,
            max_value=60,
            step=1,
            format="%d%%",
            key="sim_attrition_pct_ui",
        )
        attrition_pct = attrition_pct_ui / 100.0

        wfh_pct_ui = st.slider(
            "WFH %",
            min_value=0,
            max_value=100,
            step=1,
            format="%d%%",
            key="sim_wfh_pct_ui",
        )
        wfh_pct = wfh_pct_ui / 100.0

        st.subheader("Compensation")
        increment_pct_ui = st.slider(
            "Increment % (applied on forecast salary)",
            min_value=0,
            max_value=30,
            step=1,
            format="%d%%",
            key="sim_increment_pct_ui",
        )
        increment_pct = increment_pct_ui / 100.0
        salary_override_raw = st.number_input(
            "Avg Salary Override (INR/yr, 0 = use SARIMA)",
            min_value=0,
            key="sim_salary_override_raw",
        )
        salary_override = float(salary_override_raw) if salary_override_raw > 0 else None

        st.subheader("Budget")
        annual_budget = st.number_input(
            "Annual Budget FY26 (INR)",
            min_value=0,
            help=f"Auto-extrapolated from 3-year trend: {_fmt_inr(suggested_budget)}",
            key="sim_annual_budget",
        )

        reset = st.button("Reset to Forecast Defaults", use_container_width=True)

    if reset:
        st.session_state["sim_reset_requested"] = True
        st.rerun()

    override = {
        "closing_hc": target_hc,
        "attrition_pct": attrition_pct,
        "wfh_pct": wfh_pct,
        "increment_pct": increment_pct,
    }
    if salary_override:
        override["avg_salary"] = salary_override

    with st.spinner("Computing forecast..."):
        forecast_df = forecast_dept(
            dept=dept,
            dept_df=dept_df,
            models=models,
            rates=rates,
            horizon=12,
            override=override,
            strategy=strategy,
        )

        cf_df = run_carry_forward(
            annual_budget=float(annual_budget),
            monthly_costs=forecast_df["total_cost"].values,
            periods=forecast_df.index,
        )

    base_fc = forecast_dept(
        dept=dept,
        dept_df=dept_df,
        models=models,
        rates=rates,
        horizon=12,
        override=base_override,
        strategy=strategy,
    )
    base_annual_cost = float(base_fc["total_cost"].sum())
    sim_annual_cost = float(forecast_df["total_cost"].sum())
    delta_cost = sim_annual_cost - base_annual_cost
    delta_pct = (delta_cost / base_annual_cost * 100) if base_annual_cost else 0
    months_at_risk = int((cf_df["Flag"] == "⚠ Over Budget").sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Annual Budget FY26", _fmt_inr(annual_budget))
    k2.metric(
        "Predicted Annual Cost",
        _fmt_inr(sim_annual_cost),
        delta=f"{_fmt_inr(delta_cost)} vs base" if delta_cost != 0 else None,
        delta_color="inverse",
    )
    k3.metric("Budget Surplus / (Deficit)", _fmt_inr(annual_budget - sim_annual_cost))
    k4.metric("Months at Risk of Overrun", str(months_at_risk))

    st.subheader("Monthly Carry-Forward Schedule")

    display_cf = cf_df.copy().reset_index()
    display_cf["Period"] = display_cf["Period"].astype(str)
    display_cf["Effective Monthly Budget"] = display_cf["Effective Monthly Budget"].apply(_fmt_inr)
    display_cf["Predicted Cost"] = display_cf["Predicted Cost"].apply(_fmt_inr)
    display_cf["Pool (Start of Month)"] = display_cf["Pool (Start of Month)"].apply(_fmt_inr)
    display_cf["Pool (End of Month)"] = display_cf["Pool (End of Month)"].apply(_fmt_inr)
    display_cf["Monthly Variance"] = display_cf["Monthly Variance"].apply(_fmt_inr)

    def _colour_flag(row):
        if row["Flag"] == "⚠ Over Budget":
            style = "background-color: #c0392b; color: #ffffff; font-weight: bold"
        else:
            style = "background-color: #1a4731; color: #ffffff"
        return [style] * len(row)

    styled = display_cf.style.apply(_colour_flag, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.subheader("Monthly Cost Breakdown")
    tab_chart, tab_detail, tab_drivers = st.tabs(["Chart", "Cost Detail", "HC and Attrition"])

    with tab_chart:
        chart_df = forecast_df[["total_direct_cost", "total_indirect_cost"]].copy().reset_index()
        chart_df["period"] = chart_df["period"].astype(str)
        chart_df = chart_df.rename(columns={"total_direct_cost": "Direct Cost", "total_indirect_cost": "Indirect Cost"})
        st.bar_chart(chart_df.set_index("period"), use_container_width=True)

    with tab_detail:
        detail_cols = [
            "direct_salary_cost",
            "benefits_cost",
            "variable_pay_bonus",
            "payroll_tax",
            "travel_allowance",
            "meal_allowance",
            "overtime_cost",
            "recruitment_cost",
            "training_dev_cost",
            "it_license_cost",
            "it_equipment_cost",
            "rent_facilities_cost",
            "utilities_cost",
            "admin_overhead",
            "hr_payroll_admin",
            "learning_platform_cost",
            "employee_engagement_cost",
            "other_indirect_cost",
            "total_cost",
        ]
        avail = [c for c in detail_cols if c in forecast_df.columns]
        detail_df = forecast_df[avail].copy()
        detail_df.index = detail_df.index.astype(str)
        detail_df = detail_df.map(lambda x: f"Rs {x:,.0f}")
        st.dataframe(detail_df.T, use_container_width=True)

    with tab_drivers:
        drivers_df = forecast_df[["closing_hc", "exits", "hires", "attrition_pct", "avg_salary"]].copy()
        drivers_df["attrition_pct"] = (drivers_df["attrition_pct"] * 100).round(2).astype(str) + "%"
        drivers_df["avg_salary"] = drivers_df["avg_salary"].apply(lambda x: f"Rs {x:,.0f}")
        drivers_df.index = drivers_df.index.astype(str)
        drivers_df = drivers_df.rename(
            columns={
                "closing_hc": "Closing HC",
                "exits": "Exits",
                "hires": "Hires",
                "attrition_pct": "Attrition %",
                "avg_salary": "Avg Salary/FTE",
            }
        )
        st.dataframe(drivers_df, use_container_width=True)

    if delta_cost != 0:
        sign = "+" if delta_cost > 0 else ""
        st.info(
            f"Cost delta vs base SARIMA forecast: {sign}{_fmt_inr(delta_cost)} ({sign}{delta_pct:.1f}% annual) | "
            f"Months at risk of overrun: {months_at_risk}"
        )


st.set_page_config(page_title="Workforce Cost Forecaster", layout="wide")

if not _models_exist():
    st.error("No trained models found. Please run `python train.py` first.")
    st.stop()

data = get_data()

with st.sidebar:
    page = st.radio("View", ["Forecast Simulator", "Model Performance"], index=0)

if page == "Model Performance":
    _render_performance_dashboard()
else:
    _render_simulator(data)
