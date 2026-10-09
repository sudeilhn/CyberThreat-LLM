import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

DATA_PATH = "data/processed/multiclass_dataset.parquet"

print("=" * 65)
print("       CYBERTHREAT-LLM | CROSS VALIDATION")
print("=" * 65)

print("\n[1] Dataset yükleniyor...")

df = pd.read_parquet(DATA_PATH)

X = df.drop(columns=["AttackType"])
y = df["AttackType"]

print(f"    Kayıt   : {len(df):,}")
print(f"    Feature : {X.shape[1]}")
print(f"    Sınıf   : {y.nunique()}")

print("\n[2] Model hazırlanıyor...")

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\n[3] 5-Fold Cross-Validation başlıyor...")

scores = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "f1_macro": "f1_macro"
    },
    n_jobs=-1
)

print("\n" + "-" * 65)
print("CROSS-VALIDATION SONUÇLARI")
print("-" * 65)

for i in range(5):
    print(
        f"Fold {i + 1} | "
        f"Accuracy: {scores['test_accuracy'][i]:.4f} | "
        f"Macro F1: {scores['test_f1_macro'][i]:.4f}"
    )

print("\nOrtalama Accuracy : "
      f"{scores['test_accuracy'].mean():.4f}")

print("Ortalama Macro F1 : "
      f"{scores['test_f1_macro'].mean():.4f}")

print("F1 Std Sapma      : "
      f"{scores['test_f1_macro'].std():.4f}")

print("\n" + "=" * 65)
print("CROSS VALIDATION TAMAMLANDI")
print("=" * 65)


