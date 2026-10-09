import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("\n" + "=" * 60)
print("       CYBERTHREAT-LLM | BASELINE MODEL")
print("=" * 60)

# --------------------------------------------------
# 1. Dataset
# --------------------------------------------------

print("\n[1] Dataset yükleniyor...")

df = pd.read_parquet(
    "data/processed/bruteforce_binary.parquet"
)

print(f"    Satır sayısı : {df.shape[0]:,}")
print(f"    Sütun sayısı : {df.shape[1]}")

# --------------------------------------------------
# 2. Feature ve target
# --------------------------------------------------

print("\n[2] Feature ve target ayrılıyor...")

X = df.drop(columns=["BinaryLabel"])
y = df["BinaryLabel"]

print(f"    Feature sayısı : {X.shape[1]}")
print(f"    Target         : BinaryLabel")

# --------------------------------------------------
# 3. Train / Test Split
# --------------------------------------------------

print("\n[3] Train/Test ayrılıyor...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"    Train : {len(X_train):,}")
print(f"    Test  : {len(X_test):,}")

# --------------------------------------------------
# 4. Random Forest
# --------------------------------------------------

print("\n[4] Random Forest eğitiliyor...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("    ✓ Model eğitildi.")

# --------------------------------------------------
# 5. Prediction
# --------------------------------------------------

print("\n[5] Test verisi üzerinde tahmin yapılıyor...")

y_pred = model.predict(X_test)

# --------------------------------------------------
# 6. Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "-" * 60)
print("MODEL SONUÇLARI")
print("-" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n" + "=" * 60)
print("       BASELINE MODEL TAMAMLANDI")
print("=" * 60 + "\n")

# --------------------------------------------------
# 7. Feature Importance
# --------------------------------------------------

print("\n[7] En önemli feature'lar")
print("-" * 60)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance.head(10).to_string(index=False))


# --------------------------------------------------
# 8. Modeli kaydet
# --------------------------------------------------

print("\n[8] Model kaydediliyor...")
print("-" * 60)

model_path = "models/bruteforce_random_forest.joblib"

joblib.dump(model, model_path)

print(f"✓ Model kaydedildi: {model_path}")


