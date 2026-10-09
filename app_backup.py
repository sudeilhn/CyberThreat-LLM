from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.risk_analysis import analyze_risk

# ==================================================
# CONFIGURATION
# ==================================================

MODEL_PATH = Path("models/multiclass_random_forest.joblib")
TEST_PATH = Path("data/processed/test_dataset.parquet")

st.set_page_config(
    page_title="CyberThreat-LLM",
    page_icon="🛡️",
    layout="wide"
)

# ==================================================
# LOAD MODEL AND DATA
# ==================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_test_data():
    return pd.read_parquet(TEST_PATH)


try:
    model = load_model()
    test_df = load_test_data()
except Exception as error:
    st.error(f"Model veya test verisi yüklenemedi: {error}")
    st.stop()

FEATURES = list(model.feature_names_in_)

# ==================================================
# HEADER
# ==================================================

st.title("🛡️ CyberThreat-LLM")
st.subheader("AI-Powered Cyber Threat Detection System")

st.caption(
    "Random Forest | CICIDS2017 | SHAP Explainability | "
    "Risk Assessment"
)

st.info(
    "Research prototype: Model probabilities are not calibrated. "
    "Predictions should not be treated as verified security alerts."
)

st.divider()

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("Traffic Analysis")

source = st.sidebar.radio(
    "Select Data Source",
    ["Test Dataset", "Upload CSV"]
)

# ==================================================
# DATA SOURCE
# ==================================================

X_input = None
actual_label = None

if source == "Test Dataset":

    st.subheader("Test Dataset Analysis")

    st.write(
        "Select a record from the held-out test dataset."
    )

    record_index = st.number_input(
        "Traffic Record Index",
        min_value=0,
        max_value=len(test_df) - 1,
        value=0,
        step=1
    )

    selected = test_df.iloc[[record_index]]

    actual_label = selected["AttackType"].iloc[0]

    X_input = selected[FEATURES].copy()

else:

    st.subheader("External CSV Analysis")

    uploaded_file = st.file_uploader(
        "Upload a network traffic CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            uploaded_df = pd.read_csv(uploaded_file)

            st.write(
                f"Uploaded records: {len(uploaded_df):,}"
            )

            missing_features = [
                col for col in FEATURES
                if col not in uploaded_df.columns
            ]

            if missing_features:
                st.error(
                    "Missing required features: "
                    + ", ".join(missing_features)
                )
                st.stop()

            if uploaded_df.empty:
                st.error("The uploaded CSV is empty.")
                st.stop()

            record_index = st.number_input(
                "Select Uploaded Record",
                min_value=0,
                max_value=len(uploaded_df) - 1,
                value=0,
                step=1
            )

            selected = uploaded_df.iloc[[record_index]]

            if "AttackType" in selected.columns:
                actual_label = selected["AttackType"].iloc[0]

            X_input = selected[FEATURES].copy()

            # Validate numeric features
            X_input = X_input.apply(
                pd.to_numeric,
                errors="coerce"
            )

            X_input = X_input.replace(
                [np.inf, -np.inf],
                np.nan
            )

            if X_input.isna().any().any():
                st.error(
                    "The selected record contains missing, "
                    "invalid, or infinite feature values."
                )
                st.stop()

        except Exception as error:
            st.error(f"CSV processing error: {error}")
            st.stop()

# ==================================================
# RUN ANALYSIS
# ==================================================

if X_input is not None:

    st.divider()

    if st.button(
        "🔍 Analyze Traffic",
        type="primary"
    ):

        with st.spinner("Analyzing network traffic..."):

            prediction = model.predict(X_input)[0]

            probabilities = model.predict_proba(
                X_input
            )[0]

            predicted_index = list(
                model.classes_
            ).index(prediction)

            confidence = float(
                probabilities[predicted_index]
            )

            risk = analyze_risk(
                prediction,
                confidence
            )

        # ==========================================
        # PREDICTION RESULTS
        # ==========================================

        st.subheader("Threat Detection Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Detected Class",
            prediction
        )

        col2.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

        col3.metric(
            "Risk Level",
            risk["risk_level"]
        )

        if actual_label is not None:

            st.write(
                f"**Actual Label:** {actual_label}"
            )

            if actual_label == prediction:
                st.success("Correct prediction")
            else:
                st.warning("Incorrect prediction")

        # ==========================================
        # CLASS PROBABILITIES
        # ==========================================

        st.subheader("Class Probabilities")

        probability_df = pd.DataFrame({
            "Class": model.classes_,
            "Probability": probabilities
        })

        probability_df = probability_df.sort_values(
            "Probability",
            ascending=False
        )

        st.bar_chart(
            probability_df.set_index("Class")
        )

        # ==========================================
        # RISK ASSESSMENT
        # ==========================================

        st.subheader("Security Risk Assessment")

        if risk["review_required"]:
            st.warning(
                "Manual Review Required: "
                "Model confidence is below 80%."
            )
        else:
            st.info(
                "No low-confidence review flag. "
                "This does not guarantee the prediction is correct."
            )

        st.write(
            f"**Risk Level:** {risk['risk_level']}"
        )

        st.write(
            f"**Recommendation:** "
            f"{risk['recommendation']}"
        )

        # ==========================================
        # SHAP EXPLANATION
        # ==========================================

        st.subheader("SHAP Explainability")

        with st.spinner(
            "Calculating SHAP feature contributions..."
        ):

            try:
                explainer = shap.TreeExplainer(model)

                explanation = explainer(X_input)

                shap_values = explanation.values

                if shap_values.ndim != 3:
                    raise ValueError(
                        "Unexpected SHAP output shape."
                    )

                class_values = shap_values[
                    0, :, predicted_index
                ]

                shap_df = pd.DataFrame({
                    "Feature": FEATURES,
                    "SHAP": class_values
                })

                shap_df["AbsoluteSHAP"] = (
                    shap_df["SHAP"].abs()
                )

                shap_df = shap_df.sort_values(
                    "AbsoluteSHAP",
                    ascending=False
                )

                top_features = shap_df.head(10)

                st.write(
                    "Top 10 features influencing "
                    f"the **{prediction}** prediction:"
                )

                st.bar_chart(
                    top_features.set_index("Feature")[
                        "SHAP"
                    ]
                )

                st.dataframe(
                    top_features[
                        ["Feature", "SHAP"]
                    ],
                    hide_index=True,
                    use_container_width=True
                )

                st.caption(
                    "Positive SHAP values increase the "
                    "model output for the selected class; "
                    "negative values decrease it. "
                    "These are local model explanations, "
                    "not proof of causation."
                )

            except Exception as error:
                st.error(
                    f"SHAP analysis failed: {error}"
                )

        # ==========================================
        # TRAFFIC FEATURES
        # ==========================================

        with st.expander(
            "View All 67 Traffic Features"
        ):

            feature_df = pd.DataFrame({
                "Feature": FEATURES,
                "Value": X_input.iloc[0].values
            })

            st.dataframe(
                feature_df,
                hide_index=True,
                use_container_width=True
            )

else:
    st.info(
        "Select a traffic record or upload a CSV "
        "to begin analysis."
    )


