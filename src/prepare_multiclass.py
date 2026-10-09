
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path

# ==================================================
# CONFIGURATION
# ==================================================

RAW_DIR = Path("data/raw")
OUTPUT_PATH = Path("data/processed/multiclass_dataset.parquet")

RANDOM_STATE = 42
BATCH_SIZE = 10000

MAX_SAMPLES = {
    "Benign": 15000,
    "BruteForce": 9150,
    "DoS": 15000,
    "DDoS": 15000,
    "PortScan": 2000,
    "WebAttack": 2200,
    "Botnet": 1500
}

LABEL_MAPPING = {
    "Benign": "Benign",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "Bot": "Botnet"
}


def map_label(label):
    label = str(label)

    if label.startswith("Web Attack"):
        return "WebAttack"

    return LABEL_MAPPING.get(label)


print("=" * 65)
print("CYBERTHREAT-LLM | MULTI-CLASS DATA PREPARATION")
print("=" * 65)

files = sorted(RAW_DIR.glob("*.parquet"))

if not files:
    raise FileNotFoundError(
        "data/raw klasöründe parquet dosyası bulunamadı."
    )

rng = np.random.default_rng(RANDOM_STATE)

# ==================================================
# RESERVOIR SAMPLING
# ==================================================
# Her sınıftan belirlenen sayıda kayıt seçiyoruz.
# Tüm datasetleri aynı anda RAM'e almıyoruz.

reservoirs = {
    attack_type: []
    for attack_type in MAX_SAMPLES
}

seen_counts = {
    attack_type: 0
    for attack_type in MAX_SAMPLES
}

for file_path in files:

    print(f"\nOkunuyor: {file_path.name}")

    parquet_file = pq.ParquetFile(file_path)

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE
    ):

        df = batch.to_pandas()

        if "Label" not in df.columns:
            raise ValueError(
                f"Label sütunu bulunamadı: {file_path.name}"
            )

        df["AttackType"] = df["Label"].map(map_label)

        df = df[df["AttackType"].notna()].copy()

        df = df.drop(columns=["Label"])

        if df.empty:
            continue

        for attack_type, group in df.groupby("AttackType"):

            limit = MAX_SAMPLES[attack_type]
            reservoir = reservoirs[attack_type]

            for record in group.to_dict("records"):

                seen_counts[attack_type] += 1
                count = seen_counts[attack_type]

                if len(reservoir) < limit:
                    reservoir.append(record)

                else:
                    index = rng.integers(0, count)

                    if index < limit:
                        reservoir[index] = record

    print("  Dosya işlendi.")

# ==================================================
# COMBINE SAMPLES
# ==================================================

print("\nSeçilen kayıtlar birleştiriliyor...")

sampled_frames = []

for attack_type, records in reservoirs.items():

    if records:
        sampled_frames.append(pd.DataFrame(records))

    print(
        f"{attack_type:<12} | "
        f"Mevcut: {seen_counts[attack_type]:>8,} | "
        f"Seçilen: {len(records):>6,}"
    )

if not sampled_frames:
    raise ValueError("Hiçbir geçerli kayıt bulunamadı.")

final_df = pd.concat(
    sampled_frames,
    ignore_index=True
)

del sampled_frames
del reservoirs

final_df = final_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# ==================================================
# DATA CLEANING
# ==================================================

print("\nDataset temizleniyor...")

feature_columns = final_df.columns.drop("AttackType")

# Eksik ve sonsuz değerleri temizle
final_df[feature_columns] = final_df[
    feature_columns
].replace(
    [np.inf, -np.inf],
    np.nan
)

before_cleaning = len(final_df)

final_df = final_df.dropna(
    subset=feature_columns
)

removed_invalid = before_cleaning - len(final_df)

print(
    f"Eksik/sonsuz değerli kayıt kaldırıldı: "
    f"{removed_invalid}"
)

# Tekrarlanan kayıtları kaldır
before_duplicates = len(final_df)

final_df = final_df.drop_duplicates()

removed_duplicates = before_duplicates - len(final_df)

print(
    f"Duplicate kaldırıldı: {removed_duplicates}"
)

# Sabit özellikleri belirle
feature_columns = final_df.columns.drop("AttackType")

constant_columns = [
    column
    for column in feature_columns
    if final_df[column].nunique() <= 1
]

final_df = final_df.drop(
    columns=constant_columns
)

print(
    f"Constant feature kaldırıldı: "
    f"{len(constant_columns)}"
)

for column in constant_columns:
    print(f"  - {column}")

final_df = final_df.reset_index(drop=True)

# ==================================================
# FINAL RESULTS
# ==================================================

print("\n" + "-" * 65)
print("FINAL SINIF DAĞILIMI")
print("-" * 65)

print(final_df["AttackType"].value_counts())

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

final_df.to_parquet(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 65)
print("MULTI-CLASS DATASET HAZIR")
print("=" * 65)

print(f"Toplam kayıt: {len(final_df):,}")
print(f"Feature sayısı: {final_df.shape[1] - 1}")
print(f"Sınıf sayısı: {final_df['AttackType'].nunique()}")
print(f"Kaydedildi: {OUTPUT_PATH}")


