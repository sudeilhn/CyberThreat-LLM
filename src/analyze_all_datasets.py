import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

print("\n" + "=" * 65)
print("       CYBERTHREAT-LLM | ALL DATASETS ANALYSIS")
print("=" * 65)

parquet_files = sorted(DATA_DIR.glob("*.parquet"))

print(f"\nBulunan dataset sayısı: {len(parquet_files)}")

total_rows = 0

for index, file_path in enumerate(parquet_files, start=1):

    print("\n" + "-" * 65)
    print(f"[{index}] {file_path.name}")
    print("-" * 65)

    # Sadece Label sütununu okuyoruz
    df = pd.read_parquet(
        file_path,
        columns=["Label"]
    )

    total_rows += len(df)

    print(f"Toplam kayıt: {len(df):,}")

    print("\nSınıf dağılımı:")

    label_counts = df["Label"].value_counts()

    for label, count in label_counts.items():
        print(f"  {label:<25} {count:>10,}")

print("\n" + "=" * 65)
print("GENEL ÖZET")
print("=" * 65)

print(f"Dataset sayısı : {len(parquet_files)}")
print(f"Toplam kayıt   : {total_rows:,}")

print("\nAnaliz tamamlandı.\n")


