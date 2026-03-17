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
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.carry_forward import run_carry_forward
from src.cost_rules import load_rates
from src.data_loader import DEPARTMENTS, load_all
from src.forecaster import extrapolate_annual_budget, forecast_dept
from src.model_trainer import SARIMA_ORDERS


BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
VALIDATION_PATH = BASE_DIR / "validation_report.csv"
TRAINING_METRICS_PATH = MODELS_DIR / "training_metrics.json"

DEPT_SLUG_MAP = {
    d.lower().replace(" ", "_").replace("&", "and"): d for d in DEPARTMENTS
}


def _default_data_path(base_dir: Path) -> Path:
    from_env = os.getenv("WORKFORCE_DATA_PATH")
    if from_env:
        return Path(from_env)
    samples = sorted(base_dir.glob("cost_forecast_sample_data*.xlsx"))
    if samples:
        return samples[0]
    return base_dir / "data" / "raw" / "workforce_cost_model_v4.xlsx"


DATA_PATH = _default_data_path(BASE_DIR)


def _mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return 0.0


def _dept_artifact_stamp(dept: str) -> float:
    slug = dept.lower().replace(" ", "_").replace("&", "and")
    stamps = [_mtime(MODELS_DIR / f"per_fte_rates_{slug}.json"), _mtime(MODELS_DIR / f"series_strategy_{slug}.json")]
    for key in SARIMA_ORDERS:
        stamps.append(_mtime(MODELS_DIR / f"sarima_{key}_{slug}.pkl"))
    return max(stamps) if stamps else 0.0


@st.cache_resource(show_spinner="Loading historical data...")
def get_data(data_stamp: float):
    return load_all(DATA_PATH)


@st.cache_resource(show_spinner="Loading models...")
def get_models(dept: str, artifact_stamp: float):
    slug = dept.lower().replace(" ", "_").replace("&", "and")
    loaded = {}
    for key in SARIMA_ORDERS:
        path = MODELS_DIR / f"sarima_{key}_{slug}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                loaded[key] = pickle.load(f)
    return loaded


@st.cache_resource(show_spinner="Loading rates...")
def get_rates(dept: str, artifact_stamp: float):
    slug = dept.lower().replace(" ", "_").replace("&", "and")
    path = MODELS_DIR / f"per_fte_rates_{slug}.json"
    if path.exists():
        return load_rates(path)
    return {}


@st.cache_resource(show_spinner="Loading forecast strategy...")
def get_strategy(dept: str, artifact_stamp: float):
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
        default_round = "Validation 2026 (Train 2022-2025)" if "Validation 2026 (Train 2022-2025)" in rounds else rounds[0]
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
    target_year = 2026
    train_end_period = pd.Period(f"{target_year - 1}-12", freq="M")

    def _parse_monthly_csv(raw: str, label: str, as_pct: bool = False) -> list[float] | None:
        txt = (raw or "").strip()
        if not txt:
            return None
        parts = [p.strip() for p in txt.replace("\n", ",").split(",") if p.strip()]
        if len(parts) != 12:
            st.error(f"{label}: expected 12 comma-separated values, got {len(parts)}.")
            return None
        try:
            vals = [float(p) for p in parts]
        except Exception:
            st.error(f"{label}: values must be numeric.")
            return None
        if as_pct:
            vals = [v / 100.0 for v in vals]
        return vals

    with st.sidebar:
        st.header("Department")
        dept = st.selectbox("Select Department", DEPARTMENTS, key="sim_dept")

        st.divider()
        st.subheader("Headcount and Attrition")

        dept_df = depts_data[dept]
        hist_df = dept_df.loc[:train_end_period].copy() if (dept_df.index.min() <= train_end_period) else dept_df.copy()
        if hist_df.empty:
            hist_df = dept_df.copy()
        artifact_stamp = _dept_artifact_stamp(dept)
        models = get_models(dept, artifact_stamp)
        rates = get_rates(dept, artifact_stamp)
        strategy = get_strategy(dept, artifact_stamp)

        last_hc = int(hist_df["Closing HC"].iloc[-1])
        last_att = float(hist_df["Attrition % Annualized"].iloc[-1])
        last_wfh = float(rates.get("wfh_pct", 0.55))
        suggested_budget = extrapolate_annual_budget(hist_df)

        defaults = {
            "sim_target_hc": last_hc,
            "sim_attrition_pct_ui": int(round(last_att * 100)),
            "sim_wfh_pct_ui": int(round(last_wfh * 100)),
            "sim_increment_pct_ui": 0,
            "sim_salary_override_raw": 0,
            "sim_annual_budget": int(round(suggested_budget)),
            "sim_use_fc_budget_pool": True,
            "sim_link_band_to_total": True,
            "sim_apply_driver_overrides": False,
            "sim_override_hc_csv": "",
            "sim_override_attr_csv": "",
            "sim_override_wfh_csv": "",
            "sim_override_inc_csv": "",
            "sim_override_salary_csv": "",
            "sim_override_budget_csv": "",
            "sim_override_band1_csv": "",
            "sim_override_band2_csv": "",
            "sim_override_band3_csv": "",
            "sim_override_band4_csv": "",
            "sim_override_band5_csv": "",
            "sim_override_band6_csv": "",
        }
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
        apply_driver_overrides = st.checkbox(
            "Apply slider driver overrides",
            value=False,
            help="Off = pure model baseline. On = apply Closing HC / Attrition / WFH / Increment / Salary sliders.",
            key="sim_apply_driver_overrides",
        )
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
        use_forecast_budget_pool = st.checkbox(
            "Use forecasted monthly budget total as annual pool",
            value=True,
            help="If enabled, annual pool = sum of forecasted monthly budget.",
            key="sim_use_fc_budget_pool",
        )

        with st.expander("Advanced Monthly Driver Overrides", expanded=False):
            st.caption("Optional: 12 comma-separated values (Jan→Dec of forecast horizon).")
            st.text_area("Closing HC by month", key="sim_override_hc_csv", placeholder="e.g. 180,182,...")
            st.text_area("Attrition % by month", key="sim_override_attr_csv", placeholder="e.g. 12,12,13,...")
            st.text_area("WFH % by month", key="sim_override_wfh_csv", placeholder="e.g. 55,55,56,...")
            st.text_area("Increment % by month", key="sim_override_inc_csv", placeholder="e.g. 10,10,10,...")
            st.text_area("Avg Salary by month (INR/year)", key="sim_override_salary_csv", placeholder="e.g. 2200000,2210000,...")
            st.text_area("Monthly Budget by month (INR)", key="sim_override_budget_csv", placeholder="e.g. 18000000,17500000,...")
            st.caption("Band-wise HC (department and month-wise overrides):")
            st.text_area("Band 1 HC by month", key="sim_override_band1_csv")
            st.text_area("Band 2 HC by month", key="sim_override_band2_csv")
            st.text_area("Band 3 HC by month", key="sim_override_band3_csv")
            st.text_area("Band 4 HC by month", key="sim_override_band4_csv")
            st.text_area("Band 5 HC by month", key="sim_override_band5_csv")
            st.text_area("Band 6 HC by month", key="sim_override_band6_csv")
            link_band_to_total = st.checkbox(
                "Link band HC total to Closing HC",
                value=True,
                key="sim_link_band_to_total",
            )

        reset = st.button("Reset to Forecast Defaults", use_container_width=True)

    if reset:
        st.session_state["sim_reset_requested"] = True
        st.rerun()

    override = {}
    if apply_driver_overrides:
        override = {
            "closing_hc": target_hc,
            "attrition_pct": attrition_pct,
            "wfh_pct": wfh_pct,
            "increment_pct": increment_pct,
            "link_band_hc_to_closing_hc": st.session_state.get("sim_link_band_to_total", True),
        }
        if salary_override:
            override["avg_salary"] = salary_override

    # Advanced month-wise overrides (if provided, these supersede scalar sliders).
    adv_map = [
        ("sim_override_hc_csv", "closing_hc", "Closing HC by month", False),
        ("sim_override_attr_csv", "attrition_pct", "Attrition % by month", True),
        ("sim_override_wfh_csv", "wfh_pct", "WFH % by month", True),
        ("sim_override_inc_csv", "increment_pct", "Increment % by month", True),
        ("sim_override_salary_csv", "avg_salary", "Avg Salary by month", False),
        ("sim_override_budget_csv", "monthly_budget", "Monthly Budget by month", False),
        ("sim_override_band1_csv", "band_hc_1", "Band 1 HC by month", False),
        ("sim_override_band2_csv", "band_hc_2", "Band 2 HC by month", False),
        ("sim_override_band3_csv", "band_hc_3", "Band 3 HC by month", False),
        ("sim_override_band4_csv", "band_hc_4", "Band 4 HC by month", False),
        ("sim_override_band5_csv", "band_hc_5", "Band 5 HC by month", False),
        ("sim_override_band6_csv", "band_hc_6", "Band 6 HC by month", False),
    ]
    for state_key, override_key, label, as_pct in adv_map:
        arr = _parse_monthly_csv(st.session_state.get(state_key, ""), label, as_pct=as_pct)
        if arr is not None:
            override[override_key] = arr
    if override:
        override["link_band_hc_to_closing_hc"] = st.session_state.get("sim_link_band_to_total", True)

    with st.spinner("Computing forecast..."):
        forecast_df = forecast_dept(
            dept=dept,
            dept_df=hist_df,
            models=models,
            rates=rates,
            horizon=12,
            override=override,
            strategy=strategy,
        )

        monthly_budget_plan = None
        if "monthly_budget" in forecast_df.columns:
            mb = forecast_df["monthly_budget"].astype(float)
            if mb.notna().any():
                monthly_budget_plan = mb.ffill().bfill().values

        annual_pool_used = float(annual_budget)
        if use_forecast_budget_pool and monthly_budget_plan is not None:
            annual_pool_used = float(monthly_budget_plan.sum())

        cf_df = run_carry_forward(
            annual_budget=annual_pool_used,
            monthly_costs=forecast_df["total_cost"].values,
            periods=forecast_df.index,
            planned_monthly_budget=monthly_budget_plan,
        )

    base_fc = forecast_dept(
        dept=dept,
        dept_df=hist_df,
        models=models,
        rates=rates,
        horizon=12,
        override={},
        strategy=strategy,
    )
    base_annual_cost = float(base_fc["total_cost"].sum())
    sim_annual_cost = float(forecast_df["total_cost"].sum())
    delta_cost = sim_annual_cost - base_annual_cost
    delta_pct = (delta_cost / base_annual_cost * 100) if base_annual_cost else 0
    months_at_risk = int((cf_df["Flag"] == "⚠ Over Budget").sum())
    if "Planned Variance" in cf_df.columns:
        months_at_risk_effective = int((cf_df["Effective Flag"] == "⚠ Over Budget").sum())
    else:
        months_at_risk_effective = months_at_risk

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Annual Budget Pool", _fmt_inr(annual_pool_used))
    k2.metric(
        "Predicted Annual Cost",
        _fmt_inr(sim_annual_cost),
        delta=f"{_fmt_inr(delta_cost)} vs base" if delta_cost != 0 else None,
        delta_color="inverse",
    )
    k3.metric("Budget Surplus / (Deficit)", _fmt_inr(annual_pool_used - sim_annual_cost))
    k4.metric("Months at Risk of Overrun", str(months_at_risk))
    if "Planned Variance" in cf_df.columns:
        st.caption(
            f"Planned-budget risk months: {months_at_risk} | "
            f"Carry-forward effective risk months: {months_at_risk_effective}"
        )
    if sim_annual_cost > annual_pool_used:
        st.warning(
            "Forecast indicates structural annual deficit: total predicted cost exceeds annual budget pool. "
            "In this case, many (or all) months can show overrun even when logic is correct."
        )

    st.subheader("Monthly Carry-Forward Schedule")

    # Trend view: actual 2022-2025 and predicted 2026.
    trend_actual = dept_df.loc["2022-01":"2025-12"].copy()
    trend_df = pd.DataFrame(index=pd.period_range("2022-01", f"{target_year}-12", freq="M"))
    if "Original Budget (INR)" in trend_actual.columns:
        trend_df.loc[trend_actual.index, "Actual Budget (2022-2025)"] = trend_actual["Original Budget (INR)"].astype(float)
    if "Total Actual Cost (INR)" in trend_actual.columns:
        trend_df.loc[trend_actual.index, "Actual Cost (2022-2025)"] = trend_actual["Total Actual Cost (INR)"].astype(float)
    if "monthly_budget" in forecast_df.columns:
        trend_df.loc[forecast_df.index, "Pred Budget (2026)"] = forecast_df["monthly_budget"].astype(float)
    trend_df.loc[forecast_df.index, "Pred Cost (2026)"] = forecast_df["total_cost"].astype(float)
    st.subheader("Budget vs Cost Trend (Actual 2022-2025, Forecast 2026)")
    trend_plot = trend_df.copy().reset_index().rename(columns={"index": "Period"})
    trend_plot["Period"] = trend_plot["Period"].dt.to_timestamp()
    trend_long = trend_plot.melt(
        id_vars=["Period"],
        value_vars=[
            "Actual Budget (2022-2025)",
            "Actual Cost (2022-2025)",
            "Pred Budget (2026)",
            "Pred Cost (2026)",
        ],
        var_name="Series",
        value_name="INR",
    ).dropna(subset=["INR"])

    color_scale = alt.Scale(
        domain=[
            "Actual Budget (2022-2025)",
            "Actual Cost (2022-2025)",
            "Pred Budget (2026)",
            "Pred Cost (2026)",
        ],
        range=["#8ecbff", "#1e88ff", "#ffb6c1", "#ff3b3b"],
    )
    base_chart = alt.Chart(trend_long).encode(
        x=alt.X("Period:T", title="Month", axis=alt.Axis(format="%Y-%m")),
        y=alt.Y("INR:Q", title="INR"),
        color=alt.Color("Series:N", scale=color_scale, legend=alt.Legend(title="")),
    )
    line_layer = base_chart.mark_line(strokeWidth=2.5)
    point_layer = base_chart.mark_circle(size=45, opacity=0.95).encode(
        tooltip=[
            alt.Tooltip("Period:T", title="Month", format="%Y-%m"),
            alt.Tooltip("Series:N", title="Series"),
            alt.Tooltip("INR:Q", title="Value", format=",.0f"),
        ]
    )

    line_chart = (
        (line_layer + point_layer)
        .encode(
            x=alt.X("Period:T", title="Month", axis=alt.Axis(format="%Y-%m")),
            y=alt.Y("INR:Q", title="INR"),
        )
        .properties(height=340)
    )
    st.altair_chart(line_chart, use_container_width=True)

    display_cf = cf_df.copy().reset_index()
    display_cf["Period"] = display_cf["Period"].astype(str)
    if "Planned Monthly Budget" in display_cf.columns:
        display_cf["Planned Monthly Budget"] = display_cf["Planned Monthly Budget"].apply(_fmt_inr)
    if "Planned Variance" in display_cf.columns:
        display_cf["Planned Variance"] = display_cf["Planned Variance"].apply(_fmt_inr)
    if "Effective Variance" in display_cf.columns:
        display_cf["Effective Variance"] = display_cf["Effective Variance"].apply(_fmt_inr)
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
            "monthly_budget",
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
        base_driver_cols = ["closing_hc", "exits", "hires", "attrition_pct", "avg_salary"]
        band_driver_cols = [c for c in ["band_hc_1", "band_hc_2", "band_hc_3", "band_hc_4", "band_hc_5", "band_hc_6"] if c in forecast_df.columns]
        drivers_df = forecast_df[base_driver_cols + band_driver_cols].copy()
        drivers_df["attrition_pct"] = (drivers_df["attrition_pct"] * 100).round(2).astype(str) + "%"
        drivers_df["avg_salary"] = drivers_df["avg_salary"].apply(lambda x: f"Rs {x:,.0f}")
        drivers_df.index = drivers_df.index.astype(str)
        rename_map = {
            "closing_hc": "Closing HC",
            "exits": "Exits",
            "hires": "Hires",
            "attrition_pct": "Attrition %",
            "avg_salary": "Avg Salary/FTE",
            "band_hc_1": "Band 1 HC",
            "band_hc_2": "Band 2 HC",
            "band_hc_3": "Band 3 HC",
            "band_hc_4": "Band 4 HC",
            "band_hc_5": "Band 5 HC",
            "band_hc_6": "Band 6 HC",
        }
        drivers_df = drivers_df.rename(columns=rename_map)
        st.dataframe(drivers_df, use_container_width=True)

    if delta_cost != 0:
        sign = "+" if delta_cost > 0 else ""
        st.info(
            f"Cost delta vs base SARIMA forecast: {sign}{_fmt_inr(delta_cost)} ({sign}{delta_pct:.1f}% annual) | "
            f"Months at risk of overrun: {months_at_risk}"
        )


def _safe_mape(actual: pd.Series, pred: pd.Series) -> float:
    a = pd.to_numeric(actual, errors="coerce")
    p = pd.to_numeric(pred, errors="coerce")
    mask = a.notna() & p.notna() & (a != 0)
    if mask.sum() == 0:
        return float("nan")
    return float((((a[mask] - p[mask]).abs() / a[mask]).mean()) * 100.0)


def _render_2026_comparison(depts_data: dict[str, pd.DataFrame]) -> None:
    st.title("Actual vs Predicted (2026)")
    st.caption("Compares 2026 monthly actuals against baseline forecast built from training history through Dec-2025.")

    dept = st.selectbox("Select Department", DEPARTMENTS, key="cmp_dept")
    target_year = 2026
    train_end_period = pd.Period(f"{target_year - 1}-12", freq="M")
    dept_df = depts_data[dept]
    hist_df = dept_df.loc[:train_end_period].copy() if (dept_df.index.min() <= train_end_period) else dept_df.copy()
    actual_2026 = dept_df.loc[f"{target_year}-01":f"{target_year}-12"].copy()

    if hist_df.empty:
        st.error("No historical data available up to Dec-2025 for this department.")
        return
    if actual_2026.empty:
        st.warning("No 2026 actual data found in the selected workbook for this department.")
        return

    artifact_stamp = _dept_artifact_stamp(dept)
    models = get_models(dept, artifact_stamp)
    rates = get_rates(dept, artifact_stamp)
    strategy = get_strategy(dept, artifact_stamp)

    fc_df = forecast_dept(
        dept=dept,
        dept_df=hist_df,
        models=models,
        rates=rates,
        horizon=12,
        override={},
        strategy=strategy,
    )

    common_idx = fc_df.index.intersection(actual_2026.index)
    if len(common_idx) == 0:
        st.warning("No overlapping months between predicted and actual 2026 data.")
        return

    cmp = pd.DataFrame(index=common_idx)
    cmp["Actual Budget"] = pd.to_numeric(actual_2026.loc[common_idx, "Original Budget (INR)"], errors="coerce")
    cmp["Pred Budget"] = pd.to_numeric(fc_df.loc[common_idx, "monthly_budget"], errors="coerce")
    cmp["Budget Error"] = cmp["Pred Budget"] - cmp["Actual Budget"]
    cmp["Actual Cost"] = pd.to_numeric(actual_2026.loc[common_idx, "Total Actual Cost (INR)"], errors="coerce")
    cmp["Pred Cost"] = pd.to_numeric(fc_df.loc[common_idx, "total_cost"], errors="coerce")
    cmp["Cost Error"] = cmp["Pred Cost"] - cmp["Actual Cost"]

    budget_mape = _safe_mape(cmp["Actual Budget"], cmp["Pred Budget"])
    cost_mape = _safe_mape(cmp["Actual Cost"], cmp["Pred Cost"])
    budget_mae = float((cmp["Budget Error"].abs()).mean())
    cost_mae = float((cmp["Cost Error"].abs()).mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Budget MAPE (2026)", f"{budget_mape:.2f}%")
    c2.metric("Cost MAPE (2026)", f"{cost_mape:.2f}%")
    c3.metric("Budget MAE", _fmt_inr(budget_mae))
    c4.metric("Cost MAE", _fmt_inr(cost_mae))

    chart_df = cmp.reset_index().rename(columns={"index": "Period"})
    chart_df["Period"] = chart_df["Period"].dt.to_timestamp()
    chart_long = chart_df.melt(
        id_vars=["Period"],
        value_vars=["Actual Budget", "Pred Budget", "Actual Cost", "Pred Cost"],
        var_name="Series",
        value_name="INR",
    ).dropna(subset=["INR"])

    color_scale = alt.Scale(
        domain=["Actual Budget", "Pred Budget", "Actual Cost", "Pred Cost"],
        range=["#8ecbff", "#ffb6c1", "#1e88ff", "#ff3b3b"],
    )
    base_chart = alt.Chart(chart_long).encode(
        x=alt.X("Period:T", title="Month", axis=alt.Axis(format="%Y-%m")),
        y=alt.Y("INR:Q", title="INR"),
        color=alt.Color("Series:N", scale=color_scale, legend=alt.Legend(title="")),
    )
    line_layer = base_chart.mark_line(strokeWidth=2.5)
    point_layer = base_chart.mark_circle(size=45, opacity=0.95).encode(
        tooltip=[
            alt.Tooltip("Period:T", title="Month", format="%Y-%m"),
            alt.Tooltip("Series:N", title="Series"),
            alt.Tooltip("INR:Q", title="Value", format=",.0f"),
        ]
    )
    st.altair_chart((line_layer + point_layer).properties(height=340), use_container_width=True)

    display = cmp.copy().reset_index().rename(columns={"index": "Period"})
    display["Period"] = display["Period"].astype(str)
    for col in ["Actual Budget", "Pred Budget", "Budget Error", "Actual Cost", "Pred Cost", "Cost Error"]:
        display[col] = display[col].apply(_fmt_inr)
    st.subheader("Monthly Comparison Table")
    st.dataframe(display, hide_index=True, use_container_width=True)


st.set_page_config(page_title="Workforce Cost Forecaster", layout="wide")

if not _models_exist():
    st.error("No trained models found. Please run `python train.py` first.")
    st.stop()

data = get_data(_mtime(DATA_PATH))

with st.sidebar:
    page = st.radio("View", ["Forecast Simulator", "Actual vs Predicted 2026", "Model Performance"], index=0)

if page == "Model Performance":
    _render_performance_dashboard()
elif page == "Actual vs Predicted 2026":
    _render_2026_comparison(data)
else:
    _render_simulator(data)
