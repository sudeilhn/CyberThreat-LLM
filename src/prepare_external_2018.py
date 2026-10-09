
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ==========================================
# FILE PATHS
# ==========================================

INPUT_PATH = Path("data/external/ids2018_sample.csv")
OUTPUT_PATH = Path("data/external/ids2018_prepared.csv")
MODEL_PATH = Path("models/multiclass_random_forest.joblib")

CHUNK_SIZE = 10000
SAMPLES_PER_CLASS = 500
RANDOM_STATE = 42

# ==========================================
# COLUMN NAME MAPPING
# CICIDS2018 -> CICIDS2017
# ==========================================

COLUMN_MAPPING = {
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Fwd Packets Length Total",
    "TotLen Bwd Pkts": "Bwd Packets Length Total",
    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "Fwd IAT Tot": "Fwd IAT Total",
    "Bwd IAT Tot": "Bwd IAT Total",
    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s",
    "Bwd Pkts/s": "Bwd Packets/s",
    "Pkt Len Min": "Packet Length Min",
    "Pkt Len Max": "Packet Length Max",
    "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",
    "FIN Flag Cnt": "FIN Flag Count",
    "SYN Flag Cnt": "SYN Flag Count",
    "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count",
    "ACK Flag Cnt": "ACK Flag Count",
    "URG Flag Cnt": "URG Flag Count",
    "ECE Flag Cnt": "ECE Flag Count",
    "Pkt Size Avg": "Avg Packet Size",
    "Fwd Seg Size Avg": "Avg Fwd Segment Size",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size",
    "Subflow Fwd Pkts": "Subflow Fwd Packets",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Subflow Bwd Byts": "Subflow Bwd Bytes",
    "Init Fwd Win Byts": "Init Fwd Win Bytes",
    "Init Bwd Win Byts": "Init Bwd Win Bytes",
    "Fwd Act Data Pkts": "Fwd Act Data Packets",
}

# ==========================================
# LOAD MODEL FEATURES
# ==========================================

model = joblib.load(MODEL_PATH)
FEATURES = list(model.feature_names_in_)

print("=" * 55)
print("CYBERTHREAT-LLM - EXTERNAL DATA PREPARATION")
print("=" * 55)

print(f"Model features: {len(FEATURES)}")

# ==========================================
# VERIFY FEATURE MAPPING
# ==========================================

header = pd.read_csv(INPUT_PATH, nrows=0)
mapped_columns = header.rename(columns=COLUMN_MAPPING).columns

missing_features = [
    feature for feature in FEATURES
    if feature not in mapped_columns
]

if missing_features:
    raise ValueError(
        "Missing features after mapping: "
        + ", ".join(missing_features)
    )

if len(mapped_columns) != len(set(mapped_columns)):
    raise ValueError(
        "Duplicate column names detected after mapping."
    )

print("Feature names verified successfully.")

# ==========================================
# SAMPLE DATA IN SMALL CHUNKS
# ==========================================

rng = np.random.default_rng(RANDOM_STATE)

samples = {
    "Benign": [],
    "Infilteration": [],
}

seen_counts = {
    "Benign": 0,
    "Infilteration": 0,
}

total_rows = 0
invalid_rows = 0

print("\nReading CSV in chunks...")

for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_PATH,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):

    total_rows += len(chunk)

    chunk.columns = chunk.columns.str.strip()
    chunk = chunk.rename(columns=COLUMN_MAPPING)

    chunk["Label"] = (
        chunk["Label"].astype(str).str.strip()
    )

    chunk = chunk[
        chunk["Label"].isin(samples.keys())
    ].copy()

    if chunk.empty:
        continue

    X = chunk[FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid_mask = X.notna().all(axis=1)
    invalid_rows += int((~valid_mask).sum())

    valid_data = X.loc[valid_mask].copy()
    valid_data["OriginalLabel"] = chunk.loc[
        valid_mask, "Label"
    ].values

    for row in valid_data.to_dict("records"):

        label = row["OriginalLabel"]
        seen_counts[label] += 1

        reservoir = samples[label]

        if len(reservoir) < SAMPLES_PER_CLASS:
            reservoir.append(row)
        else:
            replacement_index = rng.integers(
                0, seen_counts[label]
            )

            if replacement_index < SAMPLES_PER_CLASS:
                reservoir[replacement_index] = row

    if chunk_number % 10 == 0:
        print(
            f"Processed approximately "
            f"{total_rows:,} rows..."
        )

# ==========================================
# SAVE PREPARED DATA
# ==========================================

all_samples = []

for label, records in samples.items():
    print(
        f"{label}: {seen_counts[label]:,} valid rows, "
        f"{len(records)} sampled"
    )

    all_samples.extend(records)

if not all_samples:
    raise ValueError(
        "No valid records found in the external dataset."
    )

prepared_df = pd.DataFrame(all_samples)

prepared_df = prepared_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# Keep all model features in the correct order.
prepared_df = prepared_df[
    FEATURES + ["OriginalLabel"]
]

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

prepared_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 55)
print("PREPARATION COMPLETE")
print("=" * 55)

print(f"Total input rows: {total_rows:,}")
print(f"Invalid numeric rows: {invalid_rows:,}")
print(f"Prepared records: {len(prepared_df):,}")
print(f"Output: {OUTPUT_PATH}")
print("\nOriginal labels:")
print(
    prepared_df["OriginalLabel"]
    .value_counts()
    .to_string()
)

print(
    "\nIMPORTANT: Infilteration is an unseen attack "
    "class. Its predictions cannot be scored as "
    "standard seven-class accuracy."
)

print(
    "Feature names have been aligned, but feature "
    "calculation compatibility is not fully verified."
)


