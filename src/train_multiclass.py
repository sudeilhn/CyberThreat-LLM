
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# ==================================================
# CONFIGURATION
# ==================================================

DATA_PATH = Path("data/processed/multiclass_dataset.parquet")
MODEL_PATH = Path("models/multiclass_random_forest.joblib")

TRAIN_PATH = Path("data/processed/train_dataset.parquet")
TEST_PATH = Path("data/processed/test_dataset.parquet")

RANDOM_STATE = 42

print("=" * 65)
print("CYBERTHREAT-LLM | MULTI-CLASS MODEL TRAINING")
print("=" * 65)

# ==================================================
# 1. LOAD DATASET
# ==================================================

print("\n[1] Dataset yükleniyor...")

df = pd.read_parquet(DATA_PATH)

print(f"Toplam kayıt : {len(df):,}")
print(f"Sütun sayısı : {df.shape[1]}")

# ==================================================
# 2. FEATURES AND TARGET
# ==================================================

print("\n[2] Feature ve target ayrılıyor...")

X = df.drop(columns=["AttackType"])
y = df["AttackType"]

print(f"Feature sayısı : {X.shape[1]}")
print(f"Sınıf sayısı   : {y.nunique()}")

# ==================================================
# 3. TRAIN / TEST SPLIT
# ==================================================

print("\n[3] Eğitim ve test verileri ayrılıyor...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Eğitim kayıtları : {len(X_train):,}")
print(f"Test kayıtları   : {len(X_test):,}")

# Test verilerini ayrıca sakla
TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)

train_df = X_train.copy()
train_df["AttackType"] = y_train

test_df = X_test.copy()
test_df["AttackType"] = y_test

train_df.to_parquet(TRAIN_PATH, index=False)
test_df.to_parquet(TEST_PATH, index=False)

print(f"Eğitim verisi kaydedildi : {TRAIN_PATH}")
print(f"Test verisi kaydedildi   : {TEST_PATH}")

# ==================================================
# 4. MODEL TRAINING
# ==================================================

print("\n[4] Random Forest eğitiliyor...")

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model başarıyla eğitildi.")

# ==================================================
# 5. EVALUATION
# ==================================================

print("\n[5] Model değerlendiriliyor...")

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_accuracy = accuracy_score(
    y_train, train_predictions
)

test_accuracy = accuracy_score(
    y_test, test_predictions
)

macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro"
)

print("\n" + "-" * 65)
print("TRAIN / TEST RESULTS")
print("-" * 65)

print(f"Train Accuracy : {train_accuracy:.4f}")
print(f"Test Accuracy  : {test_accuracy:.4f}")
print(f"Macro F1       : {macro_f1:.4f}")
print(f"Accuracy Gap   : {train_accuracy - test_accuracy:.4f}")

# ==================================================
# 6. CLASSIFICATION REPORT
# ==================================================

print("\n" + "-" * 65)
print("CLASSIFICATION REPORT")
print("-" * 65)

print(
    classification_report(
        y_test,
        test_predictions,
        digits=4,
        zero_division=0
    )
)

# ==================================================
# 7. CONFUSION MATRIX
# ==================================================

print("\n" + "-" * 65)
print("CONFUSION MATRIX")
print("-" * 65)

labels = list(model.classes_)

matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=labels
)

matrix_df = pd.DataFrame(
    matrix,
    index=labels,
    columns=labels
)

print(matrix_df.to_string())

# ==================================================
# 8. SAVE MODEL
# ==================================================

print("\n[8] Model kaydediliyor...")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print(f"Model kaydedildi: {MODEL_PATH}")

print("\n" + "=" * 65)
print("MULTI-CLASS MODEL TRAINING COMPLETED")
print("=" * 65)


