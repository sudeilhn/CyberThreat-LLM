import pandas as pd

benign_path = "data/raw/Benign-Monday-no-metadata.parquet"
bruteforce_path = "data/raw/Bruteforce-Tuesday-no-metadata.parquet"

print("\n" + "=" * 60)
print("       CYBERTHREAT-LLM | DATA EXPLORATION")
print("=" * 60)

print("\n[1] Datasetler yükleniyor...")

benign_df = pd.read_parquet(benign_path)
bruteforce_df = pd.read_parquet(bruteforce_path)

print("    ✓ Datasetler başarıyla yüklendi.")

# BENIGN
print("\n" + "-" * 60)
print("[2] BENIGN DATASET")
print("-" * 60)

print(f"Satır sayısı   : {benign_df.shape[0]:,}")
print(f"Sütun sayısı  : {benign_df.shape[1]}")

print("\nEtiket dağılımı:")
print(benign_df["Label"].value_counts().to_string())

# BRUTE FORCE
print("\n" + "-" * 60)
print("[3] BRUTEFORCE DATASET")
print("-" * 60)

print(f"Satır sayısı   : {bruteforce_df.shape[0]:,}")
print(f"Sütun sayısı  : {bruteforce_df.shape[1]}")

print("\nEtiket dağılımı:")
print(bruteforce_df["Label"].value_counts().to_string())

# GENEL ÖZET
print("\n" + "-" * 60)
print("[4] GENEL ÖZET")
print("-" * 60)

print(f"Toplam satır   : {len(benign_df) + len(bruteforce_df):,}")
print(f"Feature sayısı : {benign_df.shape[1] - 1}")
print("Hedef sütun    : Label")

print("\n" + "=" * 60)
print("       DATA EXPLORATION TAMAMLANDI")
print("=" * 60 + "\n")
