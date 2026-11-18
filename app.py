"""RetainAI customer retention analytics dashboard."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from lifelines import KaplanMeierFitter


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
PAGES = [
    "Customer Overview",
    "Individual Churn Prediction",
    "Survival Analysis",
    "Customer Lifetime Value & Segmentation",
    "Retention ROI Simulator",
]

st.set_page_config(
    page_title="RetainAI | Customer Intelligence",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #172338;
        --muted: #718096;
        --brand: #5664E9;
        --teal: #17A98B;
        --surface: #FFFFFF;
        --line: #E8ECF3;
    }
    html, body, [class*="css"] { font-family: 'Segoe UI', 'Aptos', sans-serif; }
    .stApp { background: #F6F8FC; color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(246,248,252,.86); }
    [data-testid="stSidebar"] { background: #101A2B; }
    [data-testid="stSidebar"] * { color: #E9EEFA; }
    [data-testid="stSidebar"] label { color: #AEBBD0 !important; }
    h1, h2, h3 { font-family: 'Segoe UI', 'Aptos', sans-serif; letter-spacing: -.035em; color: #172338; }
    .hero { padding: .35rem 0 1.15rem; }
    .eyebrow { color: #5664E9; text-transform: uppercase; letter-spacing: .14em;
               font-size: .73rem; font-weight: 700; margin-bottom: .35rem; }
    .hero-title { font-family: 'Segoe UI', 'Aptos', sans-serif; font-size: 2rem;
                  font-weight: 800; letter-spacing: -.045em; color: #172338; }
    .hero-subtitle { color: #718096; font-size: .98rem; margin-top: .35rem; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid var(--line); border-radius: 14px;
        padding: 18px 20px; box-shadow: 0 5px 18px rgba(27,39,65,.035);
    }
    div[data-testid="stMetricLabel"] { color: #718096; font-weight: 600; }
    div[data-testid="stMetricValue"] { color: #172338; font-family: 'Manrope',sans-serif; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line); border-radius: 14px;
    }
    .notice { border-left: 4px solid #E1A63B; background: #FFF9EC; padding: 14px 18px;
              border-radius: 8px; color: #6C5528; margin: .5rem 0 1rem; }
    .section-note { color: #718096; font-size: .9rem; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading customer and experiment data...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load project datasets and join the existing customer value/segment results."""
    telco = pd.read_csv(DATA_DIR / "processed" / "cleaned_telco.csv")
    clv = pd.read_csv(DATA_DIR / "processed" / "customer_clv.csv")
    segments = pd.read_csv(DATA_DIR / "processed" / "customer_segments.csv")
    uplift_metrics = pd.read_csv(REPORT_DIR / "uplift_model_metrics.csv")

    required_telco = {"customerID", "Churn", "tenure", "MonthlyCharges", "Contract"}
    if not required_telco.issubset(telco.columns):
        raise ValueError(f"Cleaned telecom data is missing columns: {sorted(required_telco - set(telco.columns))}")
    if telco["customerID"].duplicated().any():
        raise ValueError("customerID must be unique in cleaned_telco.csv")

    customer_value = clv[["customerID", "CLV", "ExpectedRemainingMonths"]]
    customer_segments = segments[["customerID", "Segment", "SegmentName"]]
    enriched = telco.merge(customer_value, on="customerID", how="left", validate="one_to_one")
    enriched = enriched.merge(customer_segments, on="customerID", how="left", validate="one_to_one")
    enriched["Churn"] = enriched["Churn"].astype(int)
    return telco, enriched, segments, uplift_metrics


@st.cache_resource(show_spinner="Loading trained customer models...")
def load_models() -> tuple[object, list[str], object, object]:
    """Load the saved churn, feature, Cox, and Kaplan-Meier artifacts."""
    churn_model = joblib.load(MODEL_DIR / "churn_model.pkl")
    feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")
    cox_model = joblib.load(MODEL_DIR / "cox_model.pkl")
    kaplan_meier = joblib.load(MODEL_DIR / "kaplan_meier.pkl")
    normalized_columns = [str(column).replace(" ", "_") for column in feature_columns]
    if list(churn_model.feature_names_in_) != normalized_columns:
        raise ValueError("Saved churn model and feature_columns.pkl do not agree")
    return churn_model, feature_columns, cox_model, kaplan_meier


@st.cache_resource(show_spinner="Preparing local churn explanations...")
def load_churn_explainer() -> shap.TreeExplainer:
    """Create one reusable SHAP explainer for the saved LightGBM model."""
    churn_model = joblib.load(MODEL_DIR / "churn_model.pkl")
    return shap.TreeExplainer(churn_model)


def build_churn_matrix(
    customers: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """Reproduce the training notebook's one-hot feature encoding and column order."""
    model_inputs = customers.drop(columns=["customerID", "Churn"])
    encoded = pd.get_dummies(model_inputs, drop_first=True, dtype=int)
    return encoded.reindex(columns=feature_columns, fill_value=0)


def required_project_files(
    data_dir: Path = DATA_DIR,
    model_dir: Path = MODEL_DIR,
    report_dir: Path = REPORT_DIR,
) -> list[Path]:
    """Return all required dashboard inputs that are currently missing."""
    required_paths = [
        data_dir / "processed" / "cleaned_telco.csv",
        data_dir / "processed" / "customer_clv.csv",
        data_dir / "processed" / "customer_segments.csv",
        model_dir / "churn_model.pkl",
        model_dir / "feature_columns.pkl",
        model_dir / "cox_model.pkl",
        model_dir / "kaplan_meier.pkl",
        report_dir / "uplift_model_metrics.csv",
    ]
    return [path for path in required_paths if not path.is_file()]


def render_missing_files_error(missing_paths: list[Path]) -> None:
    """Show an actionable error and stop rendering when project artifacts are absent."""
    st.error(
        "RetainAI cannot start because required project files are missing:\n\n"
        + "\n".join(f"- `{path}`" for path in missing_paths)
        + "\n\nRestore the listed data, model, or report artifacts and reload the app.",
        icon=":material/error:",
    )
    st.stop()


def calculate_customer_clv(
    monthly_charges: pd.Series | np.ndarray | float,
    expected_remaining_months: pd.Series | np.ndarray | float,
    profit_margin: float = 0.40,
) -> pd.Series | np.ndarray | float:
    """Calculate CLV with the project's monthly-revenue and margin convention."""
    return np.round(monthly_charges * expected_remaining_months * profit_margin, 2)


def predict_conditional_survival(
    customer: pd.Series,
    cox_model: object,
    max_prediction_months: int = 36,
) -> tuple[pd.Series, int, int]:
    """Predict only inside the Cox model's observed duration window."""
    current_tenure = int(customer["tenure"])
    last_observed_month = int(cox_model.baseline_survival_.index.max())
    remaining_observed_months = last_observed_month - current_tenure
    horizon = min(max_prediction_months, remaining_observed_months)
    if horizon <= 0:
        return pd.Series(dtype=float), current_tenure, last_observed_month

    times = np.arange(1, horizon + 1)
    survival = cox_model.predict_survival_function(
        build_cox_row(customer),
        times=times,
        conditional_after=[current_tenure],
    ).iloc[:, 0]
    return survival, current_tenure, last_observed_month


def simulate_roi(
    active_customers: pd.DataFrame,
    budget: float,
    discount_percent: float,
    discount_months: int,
    assumed_uplift_percent: float,
    outreach_cost: float,
    minimum_risk_percent: float,
) -> dict[str, object]:
    """Calculate a budget-capped, highest-risk-first retention scenario."""
    if budget < 0 or outreach_cost < 0:
        raise ValueError("Budget and outreach cost cannot be negative")
    if not 0 <= discount_percent <= 100:
        raise ValueError("Discount percent must be between 0 and 100")
    if not 0 <= assumed_uplift_percent <= 100:
        raise ValueError("Assumed uplift percent must be between 0 and 100")
    if discount_months < 0:
        raise ValueError("Discount duration cannot be negative")
    if not 0 <= minimum_risk_percent <= 100:
        raise ValueError("Minimum risk percent must be between 0 and 100")

    audience = active_customers.loc[
        active_customers["ChurnRisk"] >= minimum_risk_percent / 100
    ].copy()
    audience = audience.sort_values(
        "ChurnRisk", ascending=False, kind="stable"
    )
    audience["discount_cost"] = (
        audience["MonthlyCharges"] * (discount_percent / 100) * discount_months
    )
    audience["campaign_cost"] = audience["discount_cost"] + outreach_cost
    audience["cumulative_cost"] = audience["campaign_cost"].cumsum()
    selected = audience.loc[audience["cumulative_cost"] <= budget].copy()
    selected["retention_uplift"] = np.minimum(
        assumed_uplift_percent / 100, selected["ChurnRisk"]
    )
    selected["expected_savings"] = (
        selected["retention_uplift"] * selected["CLV"]
    )

    campaign_cost = float(selected["campaign_cost"].sum())
    expected_savings = float(selected["expected_savings"].sum())
    net_profit = expected_savings - campaign_cost
    return {
        "audience": audience,
        "selected": selected,
        "eligible_count": len(audience),
        "contacted": len(selected),
        "campaign_cost": campaign_cost,
        "expected_savings": expected_savings,
        "expected_retained": float(selected["retention_uplift"].sum()),
        "net_profit": net_profit,
        "roi": net_profit / campaign_cost if campaign_cost > 0 else None,
        "budget_remaining": max(0.0, budget - campaign_cost),
    }


@st.cache_data(show_spinner="Scoring customer churn risk...")
def score_customers() -> pd.DataFrame:
    """Score the saved, cleaned telecom cohort once and cache its churn probabilities."""
    telco = pd.read_csv(DATA_DIR / "processed" / "cleaned_telco.csv")
    churn_model = joblib.load(MODEL_DIR / "churn_model.pkl")
    feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")
    scores = churn_model.predict_proba(build_churn_matrix(telco, feature_columns))[:, 1]
    return pd.DataFrame({"customerID": telco["customerID"], "ChurnRisk": scores})


def make_chart(fig: go.Figure, height: int = 360) -> go.Figure:
    """Apply the shared chart styling used throughout the dashboard."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=12, r=12, t=48, b=12),
        font=dict(family="Segoe UI, Aptos, sans-serif", color="#43516A"),
        title_font=dict(family="Segoe UI, Aptos, sans-serif", size=16, color="#172338"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E8ECF3")
    fig.update_yaxes(gridcolor="#EEF1F6", zerolinecolor="#E8ECF3")
    return fig


def page_header(title: str, subtitle: str, section: str = "CUSTOMER INTELLIGENCE") -> None:
    """Render a consistent title block for each workspace page."""
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">{section}</div>
          <div class="hero-title">{title}</div>
          <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_customer_filters(customers: pd.DataFrame) -> pd.DataFrame:
    """Render shared sidebar controls and filter customer rows."""
    st.sidebar.markdown("### Customer filters")
    gender = st.sidebar.selectbox(
        "Gender", ["All"] + sorted(customers["gender"].dropna().unique().tolist())
    )
    contract = st.sidebar.selectbox(
        "Contract", ["All"] + sorted(customers["Contract"].dropna().unique().tolist())
    )
    internet = st.sidebar.selectbox(
        "Internet service",
        ["All"] + sorted(customers["InternetService"].dropna().unique().tolist()),
    )
    segment_options = ["All"] + sorted(customers["SegmentName"].dropna().unique().tolist())
    segment = st.sidebar.selectbox("Customer segment", segment_options)
    tenure_range = st.sidebar.slider(
        "Tenure range (months)",
        min_value=0,
        max_value=72,
        value=(0, 72),
        step=1,
    )
    churn_status = st.sidebar.multiselect(
        "Customer status",
        options=["Active", "Churned"],
        default=["Active", "Churned"],
    )

    filtered = customers.loc[
        customers["tenure"].between(tenure_range[0], tenure_range[1])
    ]
    if gender != "All":
        filtered = filtered.loc[filtered["gender"] == gender]
    if contract != "All":
        filtered = filtered.loc[filtered["Contract"] == contract]
    if internet != "All":
        filtered = filtered.loc[filtered["InternetService"] == internet]
    if segment != "All":
        filtered = filtered.loc[filtered["SegmentName"] == segment]
    if churn_status:
        selected_churn = [0 if status == "Active" else 1 for status in churn_status]
        filtered = filtered.loc[filtered["Churn"].isin(selected_churn)]
    else:
        filtered = filtered.iloc[0:0]
    st.sidebar.caption(f"{len(filtered):,} customers match the current filters.")
    return filtered.copy()


def render_customer_overview(
    customers: pd.DataFrame, uplift_metrics: pd.DataFrame
) -> None:
    """Render cohort KPIs, churn charts, and the isolated retail uplift benchmark."""
    page_header(
        "Customer Overview",
        "A portfolio view of customer health, churn exposure, and recurring revenue.",
    )
    if customers.empty:
        st.info("No customers match the selected filters. Adjust the sidebar to continue.")
        return

    active = customers.loc[customers["Churn"] == 0]
    churned = customers.loc[customers["Churn"] == 1]
    churn_rate = len(churned) / len(customers)
    high_risk = int((active["ChurnRisk"] >= 0.5).sum())
    expected_value_at_risk = float((active["ChurnRisk"] * active["CLV"].fillna(0)).sum())

    metric_cols = st.columns(4)
    metric_cols[0].metric("Customers in view", f"{len(customers):,}")
    metric_cols[1].metric("Observed churn rate", f"{churn_rate:.1%}")
    metric_cols[2].metric("Active, high-risk", f"{high_risk:,}")
    metric_cols[3].metric("Expected value at risk", f"${expected_value_at_risk:,.0f}")

    st.markdown("")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        churn_by_contract = (
            customers.groupby("Contract", observed=True)["Churn"]
            .mean()
            .mul(100)
            .rename("Churn rate")
            .reset_index()
            .sort_values("Churn rate", ascending=False)
        )
        fig = px.bar(
            churn_by_contract,
            x="Contract",
            y="Churn rate",
            color="Churn rate",
            color_continuous_scale=["#90D7C8", "#5664E9"],
            text=churn_by_contract["Churn rate"].map(lambda value: f"{value:.1f}%"),
            title="Observed churn by contract",
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(coloraxis_showscale=False, yaxis_title="Churn rate (%)")
        st.plotly_chart(make_chart(fig), width="stretch")
    with chart_right:
        risk = px.histogram(
            customers,
            x="ChurnRisk",
            color="Churn",
            nbins=24,
            barmode="overlay",
            opacity=0.78,
            color_discrete_map={0: "#5664E9", 1: "#F07865"},
            title="Model-estimated churn risk",
            labels={"ChurnRisk": "Predicted churn probability", "Churn": "Observed churn"},
        )
        risk.update_layout(legend_title_text="", xaxis_tickformat=".0%")
        st.plotly_chart(make_chart(risk), width="stretch")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        internet_rates = (
            customers.groupby("InternetService", observed=True)["Churn"]
            .mean()
            .mul(100)
            .rename("Churn rate")
            .reset_index()
            .sort_values("Churn rate", ascending=False)
        )
        internet_fig = px.bar(
            internet_rates,
            x="InternetService",
            y="Churn rate",
            color="InternetService",
            color_discrete_sequence=["#5664E9", "#17A98B", "#F0AA51"],
            text=internet_rates["Churn rate"].map(lambda value: f"{value:.1f}%"),
            title="Observed churn by internet service",
        )
        internet_fig.update_traces(textposition="outside")
        internet_fig.update_layout(showlegend=False, yaxis_title="Churn rate (%)")
        st.plotly_chart(make_chart(internet_fig), width="stretch")
    with chart_right:
        trend = px.scatter(
            customers.sample(min(len(customers), 1800), random_state=42),
            x="tenure",
            y="MonthlyCharges",
            color="ChurnRisk",
            color_continuous_scale=["#9BDCCA", "#5664E9", "#F07865"],
            hover_data=["customerID", "Contract"],
            title="Tenure and monthly charges",
            labels={"tenure": "Tenure (months)", "MonthlyCharges": "Monthly charges ($)"},
        )
        st.plotly_chart(make_chart(trend), width="stretch")

    st.divider()
    st.markdown("### Retail email experiment · separate research benchmark")
    st.markdown(
        '<div class="notice"><strong>Different population, different intervention.</strong> '
        "These Qini and AUUC results come from the Hillstrom retail email experiment. "
        "They are shown separately for context only and must not be used as telecom "
        "uplift estimates or customer targeting guidance.</div>",
        unsafe_allow_html=True,
    )
    uplift_left, uplift_right = st.columns([1, 1.4])
    with uplift_left:
        st.dataframe(
            uplift_metrics[
                ["treatment_group", "control_group", "test_rows", "qini", "auuc"]
            ].rename(
                columns={
                    "treatment_group": "Retail treatment",
                    "control_group": "Retail control",
                    "test_rows": "Test customers",
                    "qini": "Qini",
                    "auuc": "AUUC",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption("Held-out evaluation of randomized retail campaigns; not a telecom model.")
    with uplift_right:
        plot_path = REPORT_DIR / "uplift_evaluation_curves.png"
        if plot_path.is_file():
            st.image(str(plot_path), caption="Hillstrom retail campaign test-set curves")
        else:
            st.info("The saved Hillstrom evaluation plot is not available.")


def render_individual_prediction(
    customers: pd.DataFrame,
    telco: pd.DataFrame,
    churn_model: object,
    feature_columns: list[str],
    explainer: shap.TreeExplainer,
) -> None:
    """Show one customer's predicted risk and local TreeSHAP feature contributions."""
    page_header(
        "Individual Churn Prediction",
        "Inspect a saved LightGBM model prediction and the customer-specific factors behind it.",
        section="PREDICTIVE ANALYTICS",
    )
    if customers.empty:
        st.info("No customers match the selected filters. Adjust the sidebar to continue.")
        return

    selected_id = st.selectbox(
        "Select a customer",
        options=customers["customerID"].tolist(),
        format_func=lambda customer_id: (
            f"{customer_id}  ·  "
            f"{customers.loc[customers['customerID'].eq(customer_id), 'Contract'].iloc[0]}"
        ),
    )
    customer = telco.loc[telco["customerID"] == selected_id].iloc[0]
    customer_summary = customers.loc[customers["customerID"] == selected_id].iloc[0]
    model_row = build_churn_matrix(telco.loc[telco["customerID"] == selected_id], feature_columns)
    churn_probability = float(churn_model.predict_proba(model_row)[:, 1][0])

    metric_cols = st.columns(4)
    metric_cols[0].metric("Predicted churn probability", f"{churn_probability:.1%}")
    metric_cols[1].metric(
        "Risk tier",
        "High" if churn_probability >= 0.65 else "Medium" if churn_probability >= 0.3 else "Low",
    )
    metric_cols[2].metric("Monthly charges", f"${customer['MonthlyCharges']:,.2f}")
    metric_cols[3].metric(
        "Current CLV estimate",
        f"${customer_summary['CLV']:,.0f}" if pd.notna(customer_summary["CLV"]) else "N/A",
    )

    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.markdown("#### Customer profile")
        st.dataframe(
            pd.DataFrame(
                {
                    "Attribute": [
                        "Customer ID",
                        "Contract",
                        "Internet service",
                        "Tenure",
                        "Payment method",
                        "Number of services",
                        "Observed outcome",
                    ],
                    "Value": [
                        customer["customerID"],
                        customer["Contract"],
                        customer["InternetService"],
                        f"{int(customer['tenure'])} months",
                        customer["PaymentMethod"],
                        str(int(customer["TotalServices"])),
                        "Churned" if customer["Churn"] else "Active",
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )

    shap_values = explainer.shap_values(model_row)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    contribution_values = np.asarray(shap_values)
    if contribution_values.ndim == 3:
        contribution_values = contribution_values[0, :, -1]
    else:
        contribution_values = contribution_values.reshape(-1, len(feature_columns))[0]
    contribution_table = pd.DataFrame(
        {
            "feature": feature_columns,
            "value": model_row.iloc[0].to_numpy(),
            "contribution": contribution_values,
        }
    )
    contribution_table["feature"] = contribution_table["feature"].str.replace(
        "_", " ", regex=False
    )
    contribution_table = contribution_table.reindex(
        contribution_table["contribution"].abs().sort_values().index
    ).tail(12)
    contribution_table["direction"] = np.where(
        contribution_table["contribution"] >= 0, "Increases risk", "Reduces risk"
    )
    colors = [
        "#E56B6F" if value >= 0 else "#17A98B"
        for value in contribution_table["contribution"]
    ]
    shap_fig = go.Figure(
        go.Bar(
            x=contribution_table["contribution"],
            y=contribution_table["feature"],
            orientation="h",
            marker_color=colors,
            customdata=np.column_stack(
                [
                    contribution_table["value"].astype(str),
                    contribution_table["direction"],
                ]
            ),
            hovertemplate=(
                "%{y}<br>SHAP value: %{x:.3f}<br>Encoded feature: %{customdata[0]}"
                "<br>%{customdata[1]}<extra></extra>"
            ),
        )
    )
    shap_fig.update_layout(
        title="Top individual feature contributions",
        xaxis_title="SHAP contribution (log-odds)",
        yaxis_title="",
    )
    with detail_cols[1]:
        st.markdown("#### Why this prediction?")
        st.plotly_chart(
            make_chart(shap_fig, height=430),
            width="stretch",
        )
        st.caption(
            "Positive SHAP values push this prediction toward churn; negative values "
            "push it away. Contributions explain the model, not causation."
        )


def build_cox_row(customer: pd.Series) -> pd.DataFrame:
    """Match the six input columns used to fit the saved Cox proportional-hazards model."""
    return pd.DataFrame(
        {
            "MonthlyCharges": [float(customer["MonthlyCharges"])],
            "SeniorCitizen": [int(customer["SeniorCitizen"])],
            "Contract_One year": [int(customer["Contract"] == "One year")],
            "Contract_Two year": [int(customer["Contract"] == "Two year")],
            "InternetService_Fiber optic": [
                int(customer["InternetService"] == "Fiber optic")
            ],
            "InternetService_No": [int(customer["InternetService"] == "No")],
        }
    )


def render_survival_analysis(
    customers: pd.DataFrame,
    cox_model: object,
    kaplan_meier: object,
) -> None:
    """Show cohort Kaplan-Meier survival and an individual conditional Cox curve."""
    page_header(
        "Survival Analysis",
        "Explore observed customer retention over tenure and a customer's modeled survival outlook.",
        section="RETENTION TIMING",
    )
    if customers.empty:
        st.info("No customers match the selected filters. Adjust the sidebar to continue.")
        return

    selected_id = st.selectbox(
        "Customer for conditional survival estimate",
        customers["customerID"].tolist(),
        key="survival_customer",
    )
    selected = customers.loc[customers["customerID"] == selected_id].iloc[0]
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.markdown("#### Observed survival by contract")
        fig = go.Figure()
        observed_through = int(customers["tenure"].max())
        baseline = kaplan_meier.survival_function_.iloc[:, 0]
        baseline = baseline.loc[baseline.index <= observed_through]
        fig.add_trace(
            go.Scatter(
                x=baseline.index,
                y=baseline.values,
                mode="lines",
                name="All customers (saved KM)",
                line=dict(color="#5664E9", width=3, dash="dash"),
            )
        )
        for contract_name, cohort in customers.groupby("Contract", observed=True):
            if cohort.empty:
                continue
            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=cohort["tenure"],
                event_observed=cohort["Churn"],
                label=str(contract_name),
            )
            survival = kmf.survival_function_.iloc[:, 0]
            fig.add_trace(
                go.Scatter(
                    x=survival.index,
                    y=survival.values,
                    mode="lines",
                    name=str(contract_name),
                    line_shape="hv",
                )
            )
        fig.update_layout(
            title="Kaplan-Meier customer survival",
            xaxis_title="Tenure (months)",
            yaxis_title="Probability of remaining a customer",
            yaxis_tickformat=".0%",
            yaxis_range=[0, 1.02],
        )
        st.plotly_chart(make_chart(fig, height=420), width="stretch")
        st.caption(
            f"Kaplan-Meier estimates are observed only through month {observed_through} "
            "for this filtered cohort. No survival beyond that observation horizon is inferred."
        )

    with chart_cols[1]:
        st.markdown("#### Selected customer's future survival")
        conditional_curve, current_tenure, last_observed_month = (
            predict_conditional_survival(selected, cox_model)
        )
        if conditional_curve.empty:
            st.warning(
                f"This customer's current tenure ({current_tenure} months) is at or "
                f"beyond the model's last observed month ({last_observed_month}). "
                "The saved Cox model cannot estimate additional survival beyond its "
                "observation horizon."
            )
        else:
            personal_fig = go.Figure()
            personal_fig.add_trace(
                go.Scatter(
                    x=[0, *conditional_curve.index.tolist()],
                    y=[1.0, *conditional_curve.values.tolist()],
                    mode="lines",
                    line=dict(color="#17A98B", width=3, shape="hv"),
                    fill="tozeroy",
                    fillcolor="rgba(23,169,139,.10)",
                    name="Conditional survival",
                )
            )
            personal_fig.update_layout(
                title=f"Customer {selected_id} · from current tenure",
                xaxis_title="Additional months from today",
                yaxis_title="Conditional probability of remaining",
                yaxis_tickformat=".0%",
                yaxis_range=[0, 1.02],
            )
            st.plotly_chart(make_chart(personal_fig, height=420), width="stretch")
            st.caption(
                f"Conditional Cox estimate shown for {len(conditional_curve)} additional "
                f"month(s), capped at month {last_observed_month} (maximum 36 months). "
                "The model does not extrapolate beyond its observed follow-up window."
            )


def render_clv_segmentation(customers: pd.DataFrame) -> None:
    """Summarize the existing churn-free customer CLV and K-means segments."""
    page_header(
        "Customer Lifetime Value & Segmentation",
        "Understand the retained-customer value portfolio and the project's existing K-means segments.",
        section="VALUE & PORTFOLIO",
    )
    active = customers.loc[
        (customers["Churn"] == 0) & customers["CLV"].notna()
    ].copy()
    if active.empty:
        st.info(
            "The current filters return no retained customers with saved CLV estimates. "
            "Broaden the filters to view the value portfolio."
        )
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("Retained customers valued", f"{len(active):,}")
    metric_cols[1].metric("Total estimated CLV", f"${active['CLV'].sum():,.0f}")
    metric_cols[2].metric("Average customer CLV", f"${active['CLV'].mean():,.0f}")
    metric_cols[3].metric(
        "Expected remaining tenure",
        f"{active['ExpectedRemainingMonths'].mean():.1f} mo",
    )

    segment_summary = (
        active.groupby("SegmentName", observed=True)
        .agg(
            Customers=("customerID", "count"),
            Total_CLV=("CLV", "sum"),
            Average_CLV=("CLV", "mean"),
            Average_Monthly_Charges=("MonthlyCharges", "mean"),
        )
        .reset_index()
        .sort_values("Total_CLV", ascending=False)
    )
    chart_cols = st.columns(2)
    with chart_cols[0]:
        segment_fig = px.bar(
            segment_summary,
            x="SegmentName",
            y="Total_CLV",
            color="SegmentName",
            text=segment_summary["Total_CLV"].map(lambda value: f"${value:,.0f}"),
            title="Estimated value by customer segment",
            color_discrete_sequence=["#5664E9", "#17A98B", "#F0AA51", "#8592A8"],
        )
        segment_fig.update_traces(textposition="outside")
        segment_fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Total CLV ($)")
        st.plotly_chart(make_chart(segment_fig, height=410), width="stretch")
    with chart_cols[1]:
        clv_fig = px.histogram(
            active,
            x="CLV",
            color="SegmentName",
            nbins=30,
            barmode="overlay",
            opacity=0.8,
            title="Customer lifetime value distribution",
            labels={"CLV": "Estimated CLV ($)", "SegmentName": "Segment"},
        )
        st.plotly_chart(make_chart(clv_fig, height=410), width="stretch")

    st.markdown("#### Segment profile")
    display_summary = segment_summary.rename(
        columns={
            "SegmentName": "Segment",
            "Total_CLV": "Total CLV ($)",
            "Average_CLV": "Average CLV ($)",
            "Average_Monthly_Charges": "Avg. monthly charges ($)",
        }
    )
    st.dataframe(
        display_summary.style.format(
            {
                "Total CLV ($)": "${:,.0f}",
                "Average CLV ($)": "${:,.2f}",
                "Avg. monthly charges ($)": "${:,.2f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    top_customers = active.nlargest(10, "CLV")[
        ["customerID", "SegmentName", "tenure", "MonthlyCharges", "ExpectedRemainingMonths", "CLV"]
    ].rename(
        columns={
            "customerID": "Customer",
            "SegmentName": "Segment",
            "tenure": "Tenure (months)",
            "MonthlyCharges": "Monthly charges ($)",
            "ExpectedRemainingMonths": "Expected remaining months",
            "CLV": "Estimated CLV ($)",
        }
    )
    st.markdown("#### Highest estimated customer value")
    st.dataframe(
        top_customers.style.format(
            {
                "Monthly charges ($)": "${:,.2f}",
                "Expected remaining months": "{:.1f}",
                "Estimated CLV ($)": "${:,.2f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "CLV uses the project's saved survival analysis and 40% margin assumption. "
        "Estimates are available for the historically active customer cohort."
    )


def render_roi_simulator(customers: pd.DataFrame) -> None:
    """Simulate budget-limited telecom retention economics using explicit assumptions."""
    page_header(
        "Retention ROI Simulator",
        "Stress-test a telecom retention scenario using model risk, CLV, and editable business assumptions.",
        section="SCENARIO PLANNING",
    )
    st.markdown(
        '<div class="notice"><strong>Planning scenario, not a causal estimate.</strong> '
        "The assumed retention uplift below is supplied by the user. Hillstrom retail "
        "uplift results are not used in this telecom calculation.</div>",
        unsafe_allow_html=True,
    )
    active = customers.loc[
        (customers["Churn"] == 0) & customers["CLV"].notna()
    ].copy()
    if active.empty:
        st.info(
            "No retained customers with CLV estimates match these filters. "
            "Adjust the filters to simulate a campaign."
        )
        return

    inputs_left, inputs_right = st.columns(2)
    with inputs_left:
        budget = st.number_input(
            "Campaign budget ($)",
            min_value=0.0,
            max_value=1_000_000.0,
            value=25_000.0,
            step=1_000.0,
            help="A hard cap on outreach plus discount expenditure.",
        )
        discount_percent = st.slider(
            "Monthly discount (%)",
            min_value=0,
            max_value=50,
            value=10,
            help="Discount applied to monthly charges for the selected offer duration.",
        )
        discount_months = st.slider(
            "Discount duration (months)", min_value=1, max_value=12, value=3
        )
    with inputs_right:
        assumed_uplift = st.slider(
            "Assumed retention uplift (percentage points)",
            min_value=0,
            max_value=30,
            value=5,
            help="Scenario input: assumed absolute increase in retention probability.",
        )
        outreach_cost = st.number_input(
            "Campaign cost per contacted customer ($)",
            min_value=0.0,
            max_value=250.0,
            value=4.0,
            step=1.0,
        )
        minimum_risk = st.slider(
            "Minimum predicted churn risk (%)",
            min_value=0,
            max_value=95,
            value=30,
            help="Only active customers at or above this model risk enter the target pool.",
        )

    simulation = simulate_roi(
        active_customers=active,
        budget=budget,
        discount_percent=discount_percent,
        discount_months=discount_months,
        assumed_uplift_percent=assumed_uplift,
        outreach_cost=outreach_cost,
        minimum_risk_percent=minimum_risk,
    )
    audience = simulation["audience"]
    selected = simulation["selected"]
    contacted = int(simulation["contacted"])
    expected_savings = float(simulation["expected_savings"])
    total_campaign_cost = float(simulation["campaign_cost"])
    expected_retained = float(simulation["expected_retained"])
    net_profit = float(simulation["net_profit"])
    roi = simulation["roi"]

    metrics = st.columns(5)
    metrics[0].metric("Eligible high-risk audience", f"{len(audience):,}")
    metrics[1].metric("Customers contacted", f"{contacted:,}")
    metrics[2].metric("Expected savings", f"${expected_savings:,.0f}")
    metrics[3].metric("Campaign cost", f"${total_campaign_cost:,.0f}")
    metrics[4].metric("Net profit", f"${net_profit:,.0f}")

    outcome_cols = st.columns([1, 1.5])
    with outcome_cols[0]:
        st.metric(
            "Estimated ROI",
            f"{roi:.1%}" if roi is not None else "N/A",
            help="Net profit divided by campaign cost. N/A when campaign cost is zero.",
        )
        st.metric("Expected retained customers", f"{expected_retained:.1f}")
        st.caption(f"Budget remaining: ${float(simulation['budget_remaining']):,.0f}")
        if len(audience) == 0:
            st.info("No active customers meet the minimum predicted risk threshold.")
        elif contacted < len(audience):
            st.info(
                f"The budget covers {contacted:,} of {len(audience):,} eligible customers, "
                "prioritized by highest predicted churn risk."
            )
    with outcome_cols[1]:
        scenario = pd.DataFrame(
            {
                "Scenario component": ["Expected savings", "Campaign cost", "Net profit"],
                "Amount ($)": [expected_savings, total_campaign_cost, net_profit],
            }
        )
        scenario_fig = px.bar(
            scenario,
            x="Scenario component",
            y="Amount ($)",
            color="Scenario component",
            color_discrete_map={
                "Expected savings": "#17A98B",
                "Campaign cost": "#F0AA51",
                "Net profit": "#5664E9" if net_profit >= 0 else "#E56B6F",
            },
            text=scenario["Amount ($)"].map(lambda value: f"${value:,.0f}"),
            title="Scenario economics",
        )
        scenario_fig.update_traces(textposition="outside")
        scenario_fig.update_layout(showlegend=False)
        st.plotly_chart(make_chart(scenario_fig, height=340), width="stretch")

    st.markdown("#### Assumptions and calculation")
    st.markdown(
        f"""
        - **Audience:** {len(audience):,} retained customers in the current filters with predicted churn risk of at least {minimum_risk}%.
        - **Budget allocation:** highest-risk customers are selected first while cumulative campaign cost remains within ${budget:,.0f}.
        - **Cost per customer:** {discount_percent}% monthly discount × {discount_months} months of monthly charges, plus ${outreach_cost:,.2f} outreach cost.
        - **Expected savings:** assumed absolute retention uplift, capped at each customer's predicted churn probability, multiplied by saved CLV.
        - **Net profit:** expected savings minus campaign and discount costs. This is a scenario, not a measured causal effect.
        """
    )
    if contacted:
        with st.expander("View prioritized campaign audience"):
            table = selected[
                [
                    "customerID",
                    "Contract",
                    "ChurnRisk",
                    "CLV",
                    "campaign_cost",
                    "expected_savings",
                ]
            ].head(100).rename(
                columns={
                    "customerID": "Customer",
                    "Contract": "Contract",
                    "ChurnRisk": "Predicted churn risk",
                    "CLV": "Estimated CLV ($)",
                    "campaign_cost": "Campaign cost ($)",
                    "expected_savings": "Expected savings ($)",
                }
            )
            st.dataframe(
                table.style.format(
                    {
                        "Predicted churn risk": "{:.1%}",
                        "Estimated CLV ($)": "${:,.2f}",
                        "Campaign cost ($)": "${:,.2f}",
                        "Expected savings ($)": "${:,.2f}",
                    }
                ),
                hide_index=True,
                width="stretch",
            )


def main() -> None:
    """Build the navigation shell, shared filters, and selected analytics page."""
    st.sidebar.markdown(
        "<div style='font-family:Segoe UI,Aptos,sans-serif;font-size:1.55rem;font-weight:800;"
        "letter-spacing:-.05em;color:white;padding:.35rem 0 .1rem'>Retain<span "
        "style='color:#8290FF'>AI</span></div>"
        "<div style='font-size:.78rem;color:#91A0B7;padding-bottom:1.1rem'>"
        "CUSTOMER INTELLIGENCE</div>",
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio("Workspace", PAGES, label_visibility="collapsed")
    st.sidebar.divider()

    missing_paths = required_project_files()
    if missing_paths:
        render_missing_files_error(missing_paths)
        return
    try:
        telco, customers, _segments, uplift_metrics = load_data()
        scores = score_customers()
        customers = customers.merge(
            scores, on="customerID", how="left", validate="one_to_one"
        )
        filtered = apply_customer_filters(customers)
        churn_model, feature_columns, cox_model, kaplan_meier = load_models()
    except (
        OSError,
        ValueError,
        KeyError,
        ImportError,
        EOFError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        st.error(
            "RetainAI could not load its project data or saved models. "
            f"Check the input files and model compatibility. Details: {error}",
            icon=":material/error:",
        )
        return

    if page == "Customer Overview":
        render_customer_overview(filtered, uplift_metrics)
    elif page == "Individual Churn Prediction":
        explainer = load_churn_explainer()
        render_individual_prediction(
            filtered, telco, churn_model, feature_columns, explainer
        )
    elif page == "Survival Analysis":
        render_survival_analysis(filtered, cox_model, kaplan_meier)
    elif page == "Customer Lifetime Value & Segmentation":
        render_clv_segmentation(filtered)
    elif page == "Retention ROI Simulator":
        render_roi_simulator(filtered)

    st.sidebar.divider()
    st.sidebar.caption("Telecom analytics · Internal decision support")


if __name__ == "__main__":
    main()