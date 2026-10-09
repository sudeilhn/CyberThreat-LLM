import pandas as pd
import joblib

from pathlib import Path

# ==============================================
# CONFIGURATION
# ==============================================

MODEL_PATH = Path("models/multiclass_random_forest.joblib")
TEST_PATH = Path("data/processed/test_dataset.parquet")

RANDOM_STATE = 42
CONFIDENCE_THRESHOLD = 0.80

# ==============================================
# RISK RULES
# ==============================================

RISK_LEVELS = {
    "Benign": "Low",
    "PortScan": "Medium",
    "BruteForce": "High",
    "DoS": "High",
    "DDoS": "High",
    "WebAttack": "High",
    "Botnet": "Critical"
}

RECOMMENDATIONS = {
    "Benign": (
        "No attack detected. Continue routine monitoring."
    ),
    "PortScan": (
        "Review connection attempts and investigate "
        "possible reconnaissance activity."
    ),
    "BruteForce": (
        "Review authentication logs and consider "
        "account protection measures."
    ),
    "DoS": (
        "Inspect traffic volume and investigate"
        "potential service disruption."
    ),
    "DDoS": (
        "Investigate distributed traffic patterns "
        "and consider DDoS mitigation."
    ),
    "WebAttack": (
        "Review web application logs and investigate "
        "potential exploitation attempts."
    ),
    "Botnet": (
        "Prioritize investigation of the affected "
        "host and check for malicious activity."
    )
}

# ==============================================
# RISK ANALYSIS FUNCTION
# ==============================================

def analyze_risk(prediction, confidence):

    if prediction not in RISK_LEVELS:
        raise ValueError(
            f"Unknown attack class: {prediction}"
        )

    risk_level = RISK_LEVELS[prediction]
    recommendation = RECOMMENDATIONS[prediction]

    # Confidence is an uncalibrated model score.
    # It is NOT a percentage of security risk.
    review_required = confidence < CONFIDENCE_THRESHOLD

    if review_required:
        recommendation += (
            " Model confidence is below 80%; "
            "manual review is recommended."
        )

    return {
        "risk_level": risk_level,
        "review_required": review_required,
        "recommendation": recommendation
    }


# ==============================================
# MAIN
# ==============================================

def main():

    print("=" * 65)
    print("CYBERTHREAT-LLM | RISK ANALYSIS")
    print("=" * 65)

    print("\n[1] Loading model and test dataset...")

    model = joblib.load(MODEL_PATH)
    df = pd.read_parquet(TEST_PATH)

    # Same sample as previous prediction and SHAP
    sample = df.sample(n=1, random_state=RANDOM_STATE)

    actual_label = sample["AttackType"].iloc[0]

    X_sample = sample.drop(columns=["AttackType"])
    X_sample = X_sample[
        list(model.feature_names_in_)
    ]

    # ==========================================
    # PREDICTION
    # ==========================================

    prediction = model.predict(X_sample)[0]

    probabilities = model.predict_proba(
        X_sample
    )[0]

    predicted_index = list(
        model.classes_
    ).index(prediction)

    confidence = float(
        probabilities[predicted_index]
    )

    # ==========================================
    # RISK ASSESSMENT
    # ==========================================

    result = analyze_risk(
        prediction,
        confidence
    )

    print("\n[2] TRAFFIC PREDICTION")
    print("-" * 65)

    print(f"Actual Label : {actual_label}")
    print(f"Prediction   : {prediction}")
    print(f"Confidence   : {confidence * 100:.2f}%")

    print("\n[3] SECURITY RISK ASSESSMENT")
    print("-" * 65)

    print(f"Risk Level      : {result['risk_level']}")

    review_status = (
        "YES"
        if result["review_required"]
        else "NO"
    )

    print(f"Review Required : {review_status}")

    print("\n[4] SECURITY RECOMMENDATION")
    print("-" * 65)

    print(result["recommendation"])

    print("\n" + "=" * 65)
    print("RISK ANALYSIS COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()


