import pandas as pd
import numpy as np

DATA_PATH = "data/raw/Bruteforce-Tuesday-no-metadata.parquet"

print("\n" + "=" * 60)
print("       CYBERTHREAT-LLM | PREPROCESSING")
print("=" * 60)

# Dataset yükleme
print("\n[1] Dataset yükleniyor...")
df = pd.read_parquet(DATA_PATH)

print(f"    Satır sayısı  : {df.shape[0]:,}")
print(f"    Sütun sayısı : {df.shape[1]}")

# Sınıf dağılımı
print("\n[2] Sınıf dağılımı")
print("-" * 60)
print(df["Label"].value_counts())

# Eksik değer kontrolü
print("\n[3] Eksik değer kontrolü")
print("-" * 60)

missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("Eksik değer bulunamadı.")
else:
    print(missing)

# Infinity kontrolü
print("\n[4] Infinity kontrolü")
print("-" * 60)

numeric_df = df.select_dtypes(include=np.number)

inf_counts = np.isinf(numeric_df).sum()
inf_counts = inf_counts[inf_counts > 0]

if len(inf_counts) == 0:
    print("Infinity değeri bulunamadı.")
else:
    print(inf_counts)

# Duplicate kontrolü
print("\n[5] Duplicate satır kontrolü")
print("-" * 60)

duplicate_count = df.duplicated().sum()

print(f"Duplicate satır sayısı: {duplicate_count:,}")

# Sabit sütun kontrolü
print("\n[6] Sabit sütun kontrolü")
print("-" * 60)

constant_columns = [
    column for column in df.columns
    if df[column].nunique(dropna=False) <= 1
]

if len(constant_columns) == 0:
    print("Sabit sütun bulunamadı.")
else:
    for column in constant_columns:
        print(f"- {column}")

print("\n" + "=" * 60)
print("       ANALİZ TAMAMLANDI")
print("=" * 60 + "\n")

# Binary sınıf oluşturma
print("\n[7] Binary sınıflandırma hazırlanıyor...")
print("-" * 60)

df["BinaryLabel"] = df["Label"].apply(
    lambda x: 0 if x == "Benign" else 1
)

attack_df = df[df["BinaryLabel"] == 1]
benign_df = df[df["BinaryLabel"] == 0]

print(f"Attack kayıtları : {len(attack_df):,}")
print(f"Benign kayıtları : {len(benign_df):,}")


# Benign sınıfını saldırı sayısına eşitle
benign_sample = benign_df.sample(
    n=len(attack_df),
    random_state=42
)

balanced_df = pd.concat(
    [benign_sample, attack_df],
    ignore_index=True
)

# Satırları karıştır
balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


print("\n[8] Dengeli dataset")
print("-" * 60)

print(balanced_df["BinaryLabel"].value_counts())

print(f"\nToplam kayıt: {len(balanced_df):,}")

# Sabit sütunları kaldır
print("\n[9] Sabit sütunlar kaldırılıyor...")
print("-" * 60)

balanced_df = balanced_df.drop(columns=constant_columns)

print(f"Kaldırılan sütun sayısı : {len(constant_columns)}")
print(f"Kalan sütun sayısı      : {balanced_df.shape[1]}")


# Orijinal Label artık binary modelde kullanılmayacak
balanced_df = balanced_df.drop(columns=["Label"])


# Dataset'i kaydet
print("\n[10] Dataset kaydediliyor...")
print("-" * 60)

output_path = "data/processed/bruteforce_binary.parquet"

balanced_df.to_parquet(
    output_path,
    index=False
)

print(f"✓ Dataset kaydedildi: {output_path}")
print(f"✓ Boyut: {balanced_df.shape}")

print("\n" + "=" * 60)
print("       PREPROCESSING TAMAMLANDI")
print("=" * 60)



