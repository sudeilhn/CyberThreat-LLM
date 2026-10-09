from pathlib import Path
import hashlib

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


@st.cache_resource
def load_shap_explainer():
    return shap.TreeExplainer(load_model())


try:
    model = load_model()
    test_df = load_test_data()

except Exception as error:
    st.error(f"Model or dataset loading failed: {error}")
    st.stop()


FEATURES = list(model.feature_names_in_)
CLASSES = list(model.classes_)


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def validate_data(df):
    """Check that all required features are valid."""

    missing = [
        feature for feature in FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required features: " + ", ".join(missing)
        )

    if df.empty:
        raise ValueError("The dataset is empty.")

    X = df[FEATURES].copy()

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    invalid_rows = X.isna().any(axis=1)

    if invalid_rows.any():
        raise ValueError(
            f"{invalid_rows.sum()} records contain invalid, "
            "missing, or infinite feature values."
        )

    return X


def predict_records(X):
    """Predict multiple traffic records."""

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    confidence = probabilities.max(axis=1)

    results = pd.DataFrame({
        "Record": np.arange(len(X)),
        "Prediction": predictions,
        "Confidence (%)": np.round(
            confidence * 100, 2
        )
    })

    risk_results = [
        analyze_risk(pred, float(conf))
        for pred, conf in zip(
            predictions,
            confidence
        )
    ]

    results["Risk Level"] = [
        item["risk_level"]
        for item in risk_results
    ]

    results["Review Required"] = [
        item["review_required"]
        for item in risk_results
    ]

    results["Recommendation"] = [
        item["recommendation"]
        for item in risk_results
    ]

    return results, probabilities


# ==================================================
# SINGLE RECORD ANALYSIS
# ==================================================

def show_single_analysis(X, actual_label=None):

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    predicted_index = CLASSES.index(prediction)

    confidence = float(
        probabilities[predicted_index]
    )

    risk = analyze_risk(
        prediction,
        confidence
    )

    st.subheader("Threat Detection Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Detected Class", prediction)

    col2.metric(
        "Confidence",
        f"{confidence * 100:.2f}%"
    )

    col3.metric(
        "Risk Level",
        risk["risk_level"]
    )

    if actual_label is not None:

        st.write(f"**Actual Label:** {actual_label}")

        if str(actual_label) == str(prediction):
            st.success("Correct prediction")
        else:
            st.warning("Incorrect prediction")

    # PROBABILITIES

    st.subheader("Class Probabilities")

    probability_df = pd.DataFrame({
        "Class": CLASSES,
        "Probability": probabilities
    })

    probability_df = probability_df.sort_values(
        "Probability",
        ascending=False
    )

    st.bar_chart(
        probability_df.set_index("Class")
    )

    # RISK ASSESSMENT

    st.subheader("Security Risk Assessment")

    if risk["review_required"]:
        st.warning(
            "Manual Review Required: "
            "Model confidence is below 80%."
        )
    else:
        st.info(
            "No low-confidence review flag. "
            "This does not guarantee correctness."
        )

    st.write(
        f"**Risk Level:** {risk['risk_level']}"
    )

    st.write(
        f"**Recommendation:** {risk['recommendation']}"
    )


    # ==================================================
    # SOC SECURITY REPORT
    # ==================================================

    with st.expander("🛡️ SOC Security Report"):

        st.write("### Automated Security Analysis")

        st.write(f"**Detected Class:** {prediction}")
        st.write(f"**Model Confidence:** {confidence * 100:.2f}%")
        st.write(f"**Risk Level:** {risk['risk_level']}")

        if prediction == "Benign":
            st.write(
                "**Assessment:** The traffic record was "
                "classified as benign. This does not guarantee "
                "that the traffic is safe."
            )
        else:
            st.write(
                f"**Assessment:** The model classified this "
                f"traffic record as a potential {prediction} "
                "threat. Further investigation is recommended."
            )

        if risk["review_required"]:
            st.warning(
                "Manual review is recommended because "
                "model confidence is below 80%."
            )

        st.write(
            f"**Recommended Action:** {risk['recommendation']}"
        )

        st.caption(
            "Automatically generated from model predictions "
            "and risk rules. This is not an LLM-generated "
            "report or a confirmed security incident."
        )



    # SHAP

    st.subheader("SHAP Explainability")

    try:
        with st.spinner("Calculating SHAP values..."):

            explainer = load_shap_explainer()
            explanation = explainer(X)

            values = explanation.values

            if values.ndim != 3:
                raise ValueError(
                    f"Unexpected SHAP shape: {values.shape}"
                )

            class_values = values[
                0, :, predicted_index
            ]

            shap_df = pd.DataFrame({
                "Feature": FEATURES,
                "SHAP": class_values
            })

            shap_df["AbsoluteSHAP"] = (
                shap_df["SHAP"].abs()
            )

            top_features = shap_df.sort_values(
                "AbsoluteSHAP",
                ascending=False
            ).head(10)

            st.write(
                f"Top 10 features influencing "
                f"the **{prediction}** prediction:"
            )

            st.bar_chart(
                top_features.set_index("Feature")["SHAP"]
            )

            st.dataframe(
                top_features[["Feature", "SHAP"]],
                hide_index=True,
                use_container_width=True
            )

            st.caption(
                "Positive SHAP values increase the model "
                "output for the selected class. "
                "Negative values decrease it. "
                "SHAP explains model behavior, not causation."
            )

    except Exception as error:
        st.error(f"SHAP analysis failed: {error}")

    # FEATURES

    with st.expander("View All Traffic Features"):

        feature_df = pd.DataFrame({
            "Feature": FEATURES,
            "Value": X.iloc[0].values
        })

        st.dataframe(
            feature_df,
            hide_index=True,
            use_container_width=True
        )


# ==================================================
# BATCH ANALYSIS
# ==================================================

def show_batch_analysis(df):

    st.subheader("Batch Traffic Analysis")

    st.write(
        f"Ready to analyze **{len(df):,}** traffic records."
    )

    if st.button(
        "🚀 Analyze All Records",
        type="primary",
        key="batch_button"
    ):

        try:
            with st.spinner(
                "Analyzing all traffic records..."
            ):

                X = validate_data(df)

                results, probabilities = predict_records(X)

                if "AttackType" in df.columns:

                    actual = df["AttackType"].reset_index(
                        drop=True
                    )

                    results["Actual Label"] = actual

                    results["Correct"] = (
                        results["Prediction"].astype(str)
                        == actual.astype(str)
                    )

                for index, class_name in enumerate(CLASSES):

                    results[
                        f"Probability - {class_name} (%)"
                    ] = np.round(
                        probabilities[:, index] * 100,
                        2
                    )

                st.session_state["batch_results"] = results

                st.session_state["batch_source"] = (
                    st.session_state.get(
                        "current_source_id"
                    )
                )

        except Exception as error:
            st.error(f"Batch analysis failed: {error}")
            return

    results = st.session_state.get(
        "batch_results"
    )

    if results is None:
        return

    if st.session_state.get("batch_source") != (
        st.session_state.get("current_source_id")
    ):
        return

    # SUMMARY

    st.divider()
    st.subheader("Batch Analysis Summary")

    total_records = len(results)

    detected_threats = (
        results["Prediction"] != "Benign"
    ).sum()

    review_count = (
        results["Review Required"]
    ).sum()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Analyzed Records",
        f"{total_records:,}"
    )

    col2.metric(
        "Predicted Threats",
        f"{detected_threats:,}"
    )

    col3.metric(
        "Manual Review Flags",
        f"{review_count:,}"
    )

    st.caption(
        "Predicted threats are model classifications, "
        "not confirmed security incidents."
    )

    # CLASS DISTRIBUTION

    st.subheader("Prediction Distribution")

    distribution = (
        results["Prediction"]
        .value_counts()
        .reindex(CLASSES, fill_value=0)
    )

    st.bar_chart(distribution)

    # RISK DISTRIBUTION

    st.subheader("Risk Distribution")

    risk_distribution = (
        results["Risk Level"]
        .value_counts()
        .reindex(
            ["Low", "Medium", "High", "Critical"],
            fill_value=0
        )
    )

    st.bar_chart(risk_distribution)

    # OPTIONAL ACCURACY

    if "Correct" in results.columns:

        accuracy = results["Correct"].mean() * 100

        st.metric(
            "Accuracy on Uploaded Labeled Records",
            f"{accuracy:.2f}%"
        )

        st.caption(
            "This accuracy is calculated only against "
            "the labels provided in the uploaded file."
        )

    # RESULTS TABLE

    st.subheader("Detailed Analysis Results")

    st.dataframe(
        results,
        hide_index=True,
        use_container_width=True
    )

    # CSV EXPORT

    csv_data = results.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="📥 Download Analysis Results",
        data=csv_data,
        file_name="cyberthreat_analysis_results.csv",
        mime="text/csv"
    )


# ==================================================
# EXTERNAL EVALUATION - CICIDS2018
# ==================================================

def show_external_evaluation():

    st.header("External Dataset Evaluation — CICIDS2018")

    st.write(
        "This section presents a preliminary evaluation "
        "of the trained CICIDS2017 model using a separate "
        "CICIDS2018 dataset."
    )

    st.info(
        "Evaluation sample: 1,000 traffic records "
        "(500 Benign and 500 Infilteration)."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Benign Recognition Rate",
        "95.00%"
    )

    col2.metric(
        "Unseen Attack Flagging Rate",
        "2.00%"
    )

    col3.metric(
        "Benign False Alarm Rate",
        "5.00%"
    )

    st.divider()

    st.subheader("Actual vs. Predicted Classes")

    comparison = pd.DataFrame(
        {
            "Benign": [475, 490],
            "Botnet": [1, 2],
            "DoS": [23, 7],
            "PortScan": [1, 1]
        },
        index=[
            "Actual Benign",
            "Actual Infilteration"
        ]
    )

    st.dataframe(
        comparison,
        use_container_width=True
    )

    st.subheader("Prediction Distribution")

    st.bar_chart(comparison)

    st.subheader("Evaluation Findings")

    st.success(
        "Normal traffic recognition: "
        "475 of 500 Benign records were correctly "
        "classified (95%)."
    )

    st.warning(
        "Unseen attack detection limitation: "
        "490 of 500 Infilteration records were "
        "classified as Benign. Only 10 were "
        "flagged as another attack class (2%)."
    )

    st.write(
        "**Interpretation:** The model recognizes most "
        "normal records in this sample but struggles "
        "to flag the unseen Infilteration attack type."
    )

    st.caption(
        "Important limitation: Infilteration was not "
        "part of the training classes. Feature names "
        "were aligned between datasets, but feature "
        "extraction compatibility has not been fully "
        "verified. These results are preliminary and "
        "should not be interpreted as production "
        "detection performance."
    )


# ==================================================
# HEADER
# ==================================================

st.title("🛡️ CyberThreat-LLM")

st.subheader(
    "AI-Powered Cyber Threat Detection System"
)

st.caption(
    "Random Forest | CICIDS2017 | "
    "SHAP Explainability | Risk Assessment"
)

st.info(
    "Research prototype: Model probabilities are "
    "not calibrated. Predictions are not verified "
    "security alerts. External traffic may perform "
    "differently from CICIDS2017."
)

st.divider()


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Traffic Analysis",
        "External Evaluation (CICIDS2018)"
    ]
)


# ==================================================
# EXTERNAL EVALUATION PAGE
# ==================================================

if page == "External Evaluation (CICIDS2018)":

    show_external_evaluation()

    st.stop()


# ==================================================
# TRAFFIC ANALYSIS SIDEBAR
# ==================================================

st.sidebar.header("Traffic Analysis")

source = st.sidebar.radio(
    "Select Data Source",
    ["Test Dataset", "Upload CSV"]
)

analysis_mode = st.sidebar.radio(
    "Analysis Mode",
    [
        "Single Record Analysis",
        "Batch Analysis"
    ]
)


# ==================================================
# DATA SOURCE
# ==================================================

selected_df = None
source_id = None

if source == "Test Dataset":

    st.subheader("Test Dataset")

    selected_df = test_df
    source_id = "test_dataset"

else:

    st.subheader("External CSV Analysis")

    uploaded_file = st.file_uploader(
        "Upload a network traffic CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            file_bytes = uploaded_file.getvalue()

            source_id = hashlib.sha256(
                file_bytes
            ).hexdigest()

            selected_df = pd.read_csv(
                uploaded_file
            )

            st.success(
                f"CSV loaded: {len(selected_df):,} records."
            )

        except Exception as error:
            st.error(f"CSV loading failed: {error}")
            st.stop()


# ==================================================
# ANALYSIS
# ==================================================

if selected_df is not None:

    st.session_state["current_source_id"] = source_id

    try:
        X_all = validate_data(selected_df)

    except Exception as error:
        st.error(f"Data validation failed: {error}")
        st.stop()

    if analysis_mode == "Single Record Analysis":

        st.subheader("Single Record Analysis")

        record_index = st.number_input(
            "Select Traffic Record",
            min_value=0,
            max_value=len(selected_df) - 1,
            value=0,
            step=1
        )

        X_input = X_all.iloc[
            [record_index]
        ].copy()

        actual_label = None

        if "AttackType" in selected_df.columns:

            actual_label = selected_df[
                "AttackType"
            ].iloc[record_index]

        if st.button(
            "🔍 Analyze Traffic",
            type="primary"
        ):

            show_single_analysis(
                X_input,
                actual_label
            )

    else:

        show_batch_analysis(
            selected_df
        )

else:

    st.info(
        "Upload a CSV file or select the test dataset "
        "to begin analysis."
    )


