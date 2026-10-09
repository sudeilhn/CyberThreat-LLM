import pandas as pd
import joblib

DATA_PATH = "data/processed/multiclass_dataset.parquet"
MODEL_PATH = "models/multiclass_random_forest.joblib"

print("=" * 65)
print("       CYBERTHREAT-LLM | MODEL ANALYSIS")
print("=" * 65)

# --------------------------------------------------
# 1. DATASET VE MODEL
# --------------------------------------------------

print("\n[1] Dataset ve model yükleniyor...")

df = pd.read_parquet(DATA_PATH)
model = joblib.load(MODEL_PATH)

X = df.drop(columns=["AttackType"])

print(f"    Feature sayısı : {X.shape[1]}")
print("    ✓ Model yüklendi.")

# --------------------------------------------------
# 2. FEATURE IMPORTANCE
# --------------------------------------------------

print("\n[2] Feature Importance hesaplanıyor...")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "-" * 65)
print("TOP 15 FEATURE")
print("-" * 65)

print(
    importance_df.head(15).to_string(
        index=False
    )
)

# --------------------------------------------------
# 3. TOP FEATURE'LARIN TOPLAM ETKİSİ
# --------------------------------------------------

top_1 = importance_df.iloc[:1]["Importance"].sum()
top_5 = importance_df.iloc[:5]["Importance"].sum()
top_10 = importance_df.iloc[:10]["Importance"].sum()

print("\n" + "-" * 65)
print("FEATURE YOĞUNLUĞU")
print("-" * 65)

print(f"Top 1 feature etkisi  : {top_1:.4f}")
print(f"Top 5 feature etkisi  : {top_5:.4f}")
print(f"Top 10 feature etkisi : {top_10:.4f}")

# --------------------------------------------------
# 4. ŞÜPHELİ FEATURE İSİMLERİ
# --------------------------------------------------

print("\n" + "-" * 65)
print("BASİT LEAKAGE KONTROLÜ")
print("-" * 65)

suspicious_words = [
    "label",
    "attack",
    "class",
    "target",
    "category"
]

suspicious_features = []

for column in X.columns:

    column_lower = column.lower()

    if any(
        word in column_lower
        for word in suspicious_words
    ):
        suspicious_features.append(column)

if suspicious_features:

    print("⚠ Şüpheli feature isimleri bulundu:")

    for feature in suspicious_features:
        print(f"  - {feature}")

else:
    print("✓ Açık bir target/leakage feature ismi bulunmadı.")

print("\n" + "=" * 65)
print("MODEL ANALYSIS TAMAMLANDI")
print("=" * 65)


