
import pandas as pd
import joblib

DATA_PATH = "data/processed/test_dataset.parquet"
MODEL_PATH = "models/multiclass_random_forest.joblib"

print("=" * 60)
print("CYBERTHREAT-LLM | THREAT PREDICTION")
print("=" * 60)

# Model ve test verisini yükle
model = joblib.load(MODEL_PATH)
df = pd.read_parquet(DATA_PATH)

# Test verisinden örnek seç
sample = df.sample(n=1, random_state=42)

actual_label = sample["AttackType"].iloc[0]
X_sample = sample.drop(columns=["AttackType"])

# Tahmin
prediction = model.predict(X_sample)[0]
probabilities = model.predict_proba(X_sample)[0]

results = pd.DataFrame({
    "AttackType": model.classes_,
    "Probability": probabilities
}).sort_values(
    by="Probability",
    ascending=False
)

confidence = results.iloc[0]["Probability"] * 100

print("\n[1] TRAFFIC ANALYSIS")
print(f"Actual Label : {actual_label}")
print(f"Prediction   : {prediction}")
print(f"Confidence   : {confidence:.2f}%")

print("\n[2] CLASS PROBABILITIES")

for _, row in results.iterrows():
    print(
        f"{row['AttackType']:<12} "
        f"{row['Probability'] * 100:>7.2f}%"
    )

print("\n[3] PREDICTION RESULT")

if actual_label == prediction:
    print("CORRECT PREDICTION")
else:
    print("INCORRECT PREDICTION")

print("\n[4] TRAFFIC FEATURES")
print(X_sample.T.to_string(header=False))

print("\n" + "=" * 60)
print("PREDICTION COMPLETED")
print("=" * 60)




