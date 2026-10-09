import pandas as pd
import numpy as np
import joblib
import shap

from pathlib import Path

# ==============================================
# CONFIGURATION
# ==============================================

MODEL_PATH = Path("models/multiclass_random_forest.joblib")
TEST_PATH = Path("data/processed/test_dataset.parquet")

RANDOM_STATE = 42

print("=" * 65)
print("CYBERTHREAT-LLM | SHAP EXPLANATION")
print("=" * 65)

# ==============================================
# 1. LOAD MODEL AND DATA
# ==============================================

print("\n[1] Model ve test verisi yükleniyor...")

model = joblib.load(MODEL_PATH)
df = pd.read_parquet(TEST_PATH)

# Önceki tahminle aynı test kaydını seç
sample = df.sample(n=1, random_state=RANDOM_STATE)

actual_label = sample["AttackType"].iloc[0]
X_sample = sample.drop(columns=["AttackType"])

# Modelin eğitimde kullandığı sütun sırası
X_sample = X_sample[list(model.feature_names_in_)]

# ==============================================
# 2. MODEL PREDICTION
# ==============================================

prediction = model.predict(X_sample)[0]
probabilities = model.predict_proba(X_sample)[0]

predicted_index = list(model.classes_).index(prediction)
confidence = probabilities[predicted_index] * 100

print("\n[2] TAHMIN")

print(f"Gerçek sınıf : {actual_label}")
print(f"Tahmin       : {prediction}")
print(f"Confidence   : {confidence:.2f}%")

# ==============================================
# 3. SHAP EXPLANATION
# ==============================================

print("\n[3] SHAP hesaplanıyor...")

explainer = shap.TreeExplainer(model)
explanation = explainer(X_sample)

print(f"SHAP shape: {explanation.values.shape}")

# Multiclass SHAP:
# (sample, feature, class)
values = explanation.values

if values.ndim != 3:
    raise ValueError(
        f"Beklenmeyen SHAP boyutu: {values.shape}"
    )

class_shap_values = values[0, :, predicted_index]

# ==============================================
# 4. TOP FEATURES
# ==============================================

results = pd.DataFrame({
    "Feature": X_sample.columns,
    "FeatureValue": X_sample.iloc[0].values,
    "SHAP": class_shap_values
})

results["AbsoluteSHAP"] = np.abs(results["SHAP"])

results = results.sort_values(
    "AbsoluteSHAP",
    ascending=False
)

print("\n[4] TAHMINI EN COK ETKILEYEN 10 FEATURE")
print("-" * 65)

print(
    results[
        ["Feature", "FeatureValue", "SHAP"]
    ].head(10).to_string(index=False)
)

# ==============================================
# 5. INTERPRETATION
# ==============================================

print("\n[5] ACIKLAMA")
print("-" * 65)

for _, row in results.head(5).iterrows():

    direction = (
        "destekledi"
        if row["SHAP"] > 0
        else "azaltti"
    )

    print(
        f"{row['Feature']} ozelligi "
        f"{prediction} tahminini {direction}."
    )

print("\n" + "=" * 65)
print("SHAP ANALYSIS COMPLETED")
print("=" * 65)

