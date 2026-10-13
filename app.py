
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ----------------------------------
# 1. Page configuration
# ----------------------------------

st.set_page_config(
    page_title="Prescriber Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# Local mode uses original data when both processed files exist.
# Cloud mode uses separately generated, clearly labeled synthetic records.
DATA_DIR = Path("data")
REAL_FILES = [
    DATA_DIR / "segmented_doctors.parquet",
    DATA_DIR / "doctor_growth_2022_2023.parquet",
]
DEMO_MODE = not all(path.is_file() for path in REAL_FILES)
ACTIVE_DATA_DIR = DATA_DIR / "demo" if DEMO_MODE else DATA_DIR

if DEMO_MODE:
    demo_files = [
        ACTIVE_DATA_DIR / "segmented_doctors.parquet",
        ACTIVE_DATA_DIR / "doctor_growth_2022_2023.parquet",
    ]
    if not all(path.is_file() for path in demo_files):
        st.error(
            "Demo datasets missing. Run `python prepare_demo_data.py` "
            "and commit the generated `data/demo/` files."
        )
        st.stop()
    st.warning(
        "SYNTHETIC DEMO MODE: All prescriber records, KPIs, "
        "targeting scores, and priority tiers shown in tables "
        "are illustrative. Model evaluation tables and charts "
        "come from historical held-out test data, NOT these "
        "synthetic records. Synthetic IDs are not real NPIs."
    )

# ----------------------------------
# 2. Load dataset and model
# ----------------------------------

@st.cache_data
def load_data():
    return pd.read_parquet(
        ACTIVE_DATA_DIR / "segmented_doctors.parquet"
    )

@st.cache_resource
def load_model():
    return joblib.load(
        "models/random_forest.joblib"
    )

df = load_data()
bundle = load_model()

model = bundle["model"]
features = bundle["features"]
threshold = bundle["cost_threshold"]

# ----------------------------------
# 3. Segment names
# ----------------------------------

segment_names = {
    0: "Cost-Intensive Prescribers",
    1: "Low-Volume Prescribers",
    2: "High-Volume Prescribers",
    3: "Minimal-Activity Prescribers"
}

df["segment_name"] = df["segment"].map(segment_names)

# Model scores for exploratory use.
# These include training-set records.
df["targeting_score"] = model.predict_proba(
    df[features]
)[:, 1]

df["high_spending"] = (
    df["total_drug_cost"] >= threshold
).astype(int)

# ----------------------------------
# 4. Dashboard header
# ----------------------------------

st.title("Prescriber Intelligence Dashboard")
if DEMO_MODE:
    st.caption("Dashboard data: Synthetic demo | Historical model evaluation: CMS 2022–2023")

st.markdown(
    """
    ### Doctor Segmentation and Targeting

    **Dataset:** Medicare Part D Prescribers (2023)

    **Location:** Texas, USA

    **Specialty:** Internal Medicine

    **Models:** K-Means Clustering, Random Forest,
    Logistic Regression
    """
)

st.info(
    "This is an educational analytics project. "
    "High-spending status is defined using observed "
    "drug costs, not future prescribing potential."
)

# ----------------------------------
# 5. Sidebar filters
# ----------------------------------

st.sidebar.title("Dashboard Filters")

selected_segments = st.sidebar.multiselect(
    "Select Prescriber Segments",
    options=list(segment_names.values()),
    default=list(segment_names.values())
)

filtered = df[
    df["segment_name"].isin(selected_segments)
].copy()

# ----------------------------------
# 6. KPI cards
# ----------------------------------

st.header("Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Prescribers",
    f"{len(filtered):,}"
)

c2.metric(
    "Total Claims",
    f"{filtered['total_claims'].sum():,.0f}"
)

c3.metric(
    "Total Drug Cost",
    f"${filtered['total_drug_cost'].sum():,.0f}"
)

c4.metric(
    "High-Spending Prescribers",
    f"{filtered['high_spending'].sum():,}"
)

# ----------------------------------
# 7. Segmentation analysis
# ----------------------------------

st.header("1. Prescriber Segmentation")

segment_counts = (
    filtered["segment_name"]
    .value_counts()
)

st.subheader("Prescribers by Segment")

st.bar_chart(segment_counts)

st.subheader("Segment Profiles")

if not filtered.empty:

    profiles = filtered.groupby(
        "segment_name"
    ).agg(
        Doctors=("Prescriber NPI", "count"),
        Avg_Claims=("total_claims", "mean"),
        Avg_Drug_Cost=("total_drug_cost", "mean"),
        Avg_Unique_Drugs=("unique_drugs", "mean"),
        Avg_Cost_Per_Claim=("cost_per_claim", "mean")
    ).round(2)

    st.dataframe(
        profiles,
        use_container_width=True
    )

else:
    st.warning("Please select at least one segment.")

# ----------------------------------
# 8. Prescriber targeting
# ----------------------------------

st.header("2. Prescriber Targeting")

st.write(
    "The targeting score estimates how likely a "
    "prescriber is to belong to the observed "
    "high-spending category."
)

minimum_score = st.slider(
    "Minimum Targeting Score",
    min_value=0.0,
    max_value=1.0,
    value=0.70,
    step=0.05
)

targeted = filtered[
    filtered["targeting_score"] >= minimum_score
].copy()

targeted = targeted.sort_values(
    "targeting_score",
    ascending=False
)

st.metric(
    "Matching Prescribers",
    f"{len(targeted):,}"
)

display_columns = [
    "Prescriber NPI",
    "segment_name",
    "total_claims",
    "total_drug_cost",
    "unique_drugs",
    "targeting_score"
]

st.dataframe(
    targeted[display_columns].head(100),
    use_container_width=True,
    hide_index=True
)

# ----------------------------------
# 9. Model evaluation
# ----------------------------------

st.header("3. Model Performance")

results = pd.read_csv(
    "models/model_results.csv"
)

st.subheader("Model Comparison")

st.dataframe(
    results.round(4),
    use_container_width=True,
    hide_index=True
)

st.subheader("ROC Curve")

st.image(
    "charts/roc_curve.png",
    use_container_width=True
)

st.subheader("Confusion Matrix")

st.image(
    "charts/confusion_matrix.png",
    use_container_width=True
)

st.subheader("Feature Importance")

st.image(
    "charts/feature_importance.png",
    use_container_width=True
)

# ----------------------------------
# 10. Clustering visualizations
# ----------------------------------

st.header("4. Clustering Analysis")

st.subheader("PCA Visualization")

st.image(
    "charts/pca_segments.png",
    use_container_width=True
)

st.subheader("Segment Profile Comparison")

st.image(
    "charts/segment_profiles.png",
    use_container_width=True
)

# ----------------------------------
# 11. Export targeting results
# ----------------------------------

st.header("5. Download Targeting Results")

export_columns = [
    "Prescriber NPI",
    "segment_name",
    "total_claims",
    "unique_drugs",
    "targeting_score"
]

csv_data = targeted[
    export_columns
].to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name="prescriber_targeting_results.csv",
    mime="text/csv"
)

# ----------------------------------
# 12. Footer
# ----------------------------------

st.divider()

st.caption(
    "Source: CMS Medicare Part D Prescriber data, "
    "2023 Texas subset, obtained through a "
    "public Hugging Face mirror. "
    "Targeting scores are exploratory and "
    "are not validated future-spending forecasts."
)

# ----------------------------------
# 13. Live Doctor Prediction
# ----------------------------------

st.divider()
st.header("6. Live Doctor Prediction")

st.write(
    "Enter a doctor's prescribing activity to "
    "estimate their probability of belonging "
    "to the high-spending prescriber group."
)

with st.form("live_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        total_claims = st.number_input(
            "Total Claims",
            min_value=1,
            value=1200,
            step=100
        )

        total_fills = st.number_input(
            "Total 30-Day Fills",
            min_value=0.0,
            value=1500.0,
            step=100.0
        )

        total_days_supply = st.number_input(
            "Total Days Supply",
            min_value=0.0,
            value=45000.0,
            step=1000.0
        )

    with col2:
        unique_drugs = st.number_input(
            "Number of Unique Drugs",
            min_value=1,
            value=30,
            step=1
        )

        drug_records = st.number_input(
            "Number of Drug Records",
            min_value=1,
            value=40,
            step=1
        )

    submitted = st.form_submit_button(
        "Predict High-Spending Status",
        type="primary"
    )

if submitted:

    if unique_drugs > drug_records:
        st.error(
            "Unique drugs cannot exceed the "
            "number of drug records."
        )

    else:
        # Derived features from the original pipeline
        days_per_claim = (
            total_days_supply / total_claims
        )

        fills_per_claim = (
            total_fills / total_claims
        )

        input_data = pd.DataFrame([{
            "total_claims": total_claims,
            "total_fills": total_fills,
            "total_days_supply": total_days_supply,
            "unique_drugs": unique_drugs,
            "drug_records": drug_records,
            "days_per_claim": days_per_claim,
            "fills_per_claim": fills_per_claim
        }])

        # Preserve exact training feature order
        input_data = input_data[features]

        probability = model.predict_proba(
            input_data
        )[0, 1]

        prediction = model.predict(
            input_data
        )[0]

        st.subheader("Prediction Result")

        c1, c2 = st.columns(2)

        c1.metric(
            "High-Spending Model Score",
            f"{probability * 100:.2f}%"
        )

        c2.metric(
            "Predicted Category",
            (
                "High-Spending"
                if prediction == 1
                else "Other Prescriber"
            )
        )

        st.progress(float(probability))

        st.caption(
            "Prediction uses the saved Random Forest "
            "model. The score is a model estimate, "
            "not a verified future-spending probability "
            "or a clinical recommendation."
        )

# ----------------------------------
# 14. Future Growth Prediction
# ----------------------------------

st.divider()
st.header("7. Rising Prescriber Prediction")

st.markdown(
    """
    Predict which prescribers are likely to increase
    their total claims by at least **20% year-over-year**.

    The model uses **2022 prescribing features** to
    predict **2023 growth**.
    """
)

@st.cache_data
def load_growth_data():
    return pd.read_parquet(
        ACTIVE_DATA_DIR / "doctor_growth_2022_2023.parquet"
    )

@st.cache_resource
def load_growth_model():
    return joblib.load(
        "models/growth_random_forest.joblib"
    )

growth_df = load_growth_data()
growth_bundle = load_growth_model()

growth_model = growth_bundle["model"]
growth_features = growth_bundle["features"]
growth_log_features = growth_bundle["log_features"]

# Apply same transformation used during training
growth_X = growth_df[growth_features].copy()

growth_X[growth_log_features] = np.log1p(
    growth_X[growth_log_features].clip(lower=0)
)

growth_df["growth_score"] = (
    growth_model.predict_proba(growth_X)[:, 1]
)

# Assign priority tiers by score ranking
growth_df = growth_df.sort_values(
    "growth_score",
    ascending=False
).reset_index(drop=True)

n_doctors = len(growth_df)

growth_df["priority_tier"] = "Tier 3"

growth_df.loc[
    :int(n_doctors * 0.10) - 1,
    "priority_tier"
] = "Tier 1"

growth_df.loc[
    int(n_doctors * 0.10):
    int(n_doctors * 0.30) - 1,
    "priority_tier"
] = "Tier 2"

# ----------------------------------
# Growth model performance
# ----------------------------------

st.subheader("Growth Model Performance")

growth_results = pd.read_csv(
    "models/growth_model_results.csv"
)

st.dataframe(
    growth_results.round(4),
    use_container_width=True,
    hide_index=True
)

st.image(
    "charts/growth_lift_deciles.png",
    caption="Lift by prediction decile (held-out test set)",
    use_container_width=True
)

# ----------------------------------
# Priority targeting list
# ----------------------------------

st.subheader("Sales Priority Tiers")

st.markdown(
    """
    - **Tier 1:** Top 10% by predicted growth score
    - **Tier 2:** Next 20%
    - **Tier 3:** Remaining 70%

    These tiers are illustrative targeting rules,
    not validated sales recommendations.
    """
)

selected_tiers = st.multiselect(
    "Filter by priority tier",
    ["Tier 1", "Tier 2", "Tier 3"],
    default=["Tier 1"]
)

priority_df = growth_df[
    growth_df["priority_tier"].isin(selected_tiers)
].copy()

c1, c2, c3 = st.columns(3)

c1.metric(
    "Selected Prescribers",
    f"{len(priority_df):,}"
)

c2.metric(
    "Average Growth Score",
    (
        f"{priority_df['growth_score'].mean():.1%}"
        if not priority_df.empty else "N/A"
    )
)

c3.metric(
    "Observed Rising Prescribers",
    f"{priority_df['rising_prescriber'].sum():,}"
)

st.caption(
    "Observed rising status uses 2023 outcomes and "
    "is shown for retrospective analysis only. "
    "Growth scores for this full dataset include "
    "training-set doctors, so they are not "
    "independent evaluation results."
)

priority_columns = [
    "Prescriber NPI",
    "priority_tier",
    "total_claims_2022",
    "total_claims_2023",
    "growth_score",
    "rising_prescriber"
]

st.dataframe(
    priority_df[priority_columns].head(200),
    use_container_width=True,
    hide_index=True
)

# ----------------------------------
# CSV export
# ----------------------------------

st.download_button(
    "Download Priority List (CSV)",
    data=priority_df[
        priority_columns
    ].to_csv(index=False).encode("utf-8"),
    file_name="growth_priority_list.csv",
    mime="text/csv"
)

# ----------------------------------
# Live growth prediction
# ----------------------------------

st.subheader("Live Growth Prediction")

st.write(
    "Enter 2022 prescribing values to estimate "
    "whether claims will grow by at least 20% "
    "in the following year."
)

with st.form("growth_prediction_form"):

    a, b = st.columns(2)

    with a:
        claims_2022 = st.number_input(
            "2022 Total Claims",
            min_value=1,
            value=1000
        )

        cost_2022 = st.number_input(
            "2022 Total Drug Cost ($)",
            min_value=0.0,
            value=100000.0
        )

        fills_2022 = st.number_input(
            "2022 Total 30-Day Fills",
            min_value=0.0,
            value=1200.0
        )

        days_2022 = st.number_input(
            "2022 Total Days Supply",
            min_value=0.0,
            value=36000.0
        )

    with b:
        drugs_2022 = st.number_input(
            "2022 Unique Drugs",
            min_value=1,
            value=30
        )

        records_2022 = st.number_input(
            "2022 Drug Records",
            min_value=1,
            value=40
        )

    predict_growth = st.form_submit_button(
        "Predict Next-Year Growth",
        type="primary"
    )

if predict_growth:

    if drugs_2022 > records_2022:
        st.error(
            "Unique drugs cannot exceed drug records."
        )

    else:
        values = {
            "total_claims_2022": claims_2022,
            "total_cost_2022": cost_2022,
            "total_fills_2022": fills_2022,
            "total_days_supply_2022": days_2022,
            "unique_drugs_2022": drugs_2022,
            "drug_records_2022": records_2022,
            "cost_per_claim_2022": (
                cost_2022 / claims_2022
            ),
            "days_per_claim_2022": (
                days_2022 / claims_2022
            ),
            "fills_per_claim_2022": (
                fills_2022 / claims_2022
            )
        }

        live_X = pd.DataFrame([values])[
            growth_features
        ]

        live_X[growth_log_features] = np.log1p(
            live_X[growth_log_features].clip(lower=0)
        )

        growth_probability = (
            growth_model.predict_proba(live_X)[0, 1]
        )

        growth_prediction = (
            growth_model.predict(live_X)[0]
        )

        st.metric(
            "Rising Prescriber Model Score",
            f"{growth_probability:.2%}"
        )

        st.success(
            "Predicted: Rising Prescriber"
            if growth_prediction == 1
            else "Predicted: Non-Rising Prescriber"
        )

        st.progress(float(growth_probability))

        st.caption(
            "This model estimates the probability of "
            "at least 20% next-year claims growth for "
            "a continuing prescriber. Results are "
            "exploratory, not validated forecasts."
        )

