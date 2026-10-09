import pandas as pd
import numpy as np

DATA_PATH = "data/processed/multiclass_dataset.parquet"

print("=" * 65)
print("       CYBERTHREAT-LLM | MULTI-CLASS DATA CHECK")
print("=" * 65)

print("\n[1] Dataset yükleniyor...")

df = pd.read_parquet(DATA_PATH)

print(f"    Satır sayısı : {len(df):,}")
print(f"    Sütun sayısı : {df.shape[1]}")

print("\n[2] Sınıf dağılımı...")

print(df["AttackType"].value_counts())

# Target dışındaki feature'lar
X = df.drop(columns=["AttackType"])

print("\n[3] Eksik değer kontrolü...")

missing = X.isnull().sum().sum()

print(f"    NaN sayısı : {missing:,}")

print("\n[4] Infinity kontrolü...")

numeric_X = X.select_dtypes(include=[np.number])

inf_count = np.isinf(numeric_X).sum().sum()

print(f"    Infinity sayısı : {inf_count:,}")

print("\n[5] Constant feature kontrolü...")

constant_columns = [
    column
    for column in X.columns
    if X[column].nunique() <= 1
]

print(f"    Constant feature sayısı : {len(constant_columns)}")

for column in constant_columns:
    print(f"    - {column}")

print("\n[6] Duplicate kontrolü...")

duplicates = df.duplicated().sum()

print(f"    Duplicate kayıt : {duplicates:,}")

print("\n" + "=" * 65)
print("DATA CHECK TAMAMLANDI")
print("=" * 65)


