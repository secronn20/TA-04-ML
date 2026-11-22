
import pandas as pd
import numpy as np

from config import (
    PROCESSED_DATA_DIR
)


# ==========================================
# KONFIGURASI KOLOM
# ==========================================

INPUT_FILE = (
    PROCESSED_DATA_DIR /
    "air_quality_12_stations.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR /
    "air_quality_cleaned.csv"
)

REPORT_FILE = (
    PROCESSED_DATA_DIR /
    "missing_values_report.csv"
)

NUMERIC_FEATURES = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM"
]

TARGET_COLUMN = "PM2.5"


# ==========================================
# 1. MEMBACA DATASET
# ==========================================

def load_dataset():

    print("\n[1] MEMBACA DATASET")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    print(f"Jumlah data : {len(df):,}")
    print(f"Jumlah kolom: {len(df.columns)}")

    return df


# ==========================================
# 2. ANALISIS MISSING VALUES
# ==========================================

def analyze_missing_values(df):

    print("\n[2] ANALISIS MISSING VALUES")

    missing = pd.DataFrame({
        "Missing Values": df.isnull().sum(),
        "Persentase (%)": (
            df.isnull().mean() * 100
        ).round(2)
    })

    print(missing)

    missing.to_csv(REPORT_FILE)

    return missing


# ==========================================
# 3. MENGHAPUS DUPLIKASI
# ==========================================

def remove_duplicates(df):

    print("\n[3] MENGHAPUS DATA DUPLIKAT")

    before = len(df)

    df = df.drop_duplicates(
        subset=["station", "timestamp"]
    )

    after = len(df)

    print(f"Sebelum : {before:,}")
    print(f"Sesudah : {after:,}")
    print(f"Dihapus : {before - after:,}")

    return df


# ==========================================
# 4. PEMBERSIHAN DATA
# ==========================================

def clean_data(df):

    print("\n[4] MEMBERSIHKAN DATA")

    # Memastikan fitur numerik bertipe angka
    for col in NUMERIC_FEATURES:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Mengubah nilai tak hingga menjadi NaN
    df[NUMERIC_FEATURES] = (
        df[NUMERIC_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
    )

    # Mengurutkan data berdasarkan stasiun dan waktu
    df = df.sort_values(
        ["station", "timestamp"]
    ).reset_index(drop=True)

    # Mengisi missing values per stasiun
    df[NUMERIC_FEATURES] = (
        df.groupby("station")[NUMERIC_FEATURES]
        .transform(
            lambda group: group.ffill().bfill()
        )
    )

    # Memastikan tidak ada target yang kosong
    df = df.dropna(
        subset=[TARGET_COLUMN]
    )

    print(
        "Missing values setelah cleaning:",
        df[NUMERIC_FEATURES].isnull().sum().sum()
    )

    return df


# ==========================================
# 5. MEMILIH KOLOM
# ==========================================

def select_features(df):

    print("\n[5] MEMILIH FITUR MODEL")

    selected_columns = [
        "timestamp",
        "station"
    ] + NUMERIC_FEATURES

    df = df[selected_columns].copy()

    print("Fitur yang digunakan:")

    for col in NUMERIC_FEATURES:
        print(f"- {col}")

    print(f"\nTarget prediksi: {TARGET_COLUMN}")

    return df


# ==========================================
# 6. MENYIMPAN DATASET
# ==========================================

def save_dataset(df):

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n[6] DATASET BERHASIL DISIMPAN")
    print(f"Lokasi: {OUTPUT_FILE}")


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    print("=" * 60)
    print("DATA PREPROCESSING - BEIJING AIR QUALITY")
    print("=" * 60)

    # Membaca dataset
    df = load_dataset()

    # Missing values sebelum cleaning
    analyze_missing_values(df)

    # Menghapus duplikasi
    df = remove_duplicates(df)

    # Cleaning data
    df = clean_data(df)

    # Memilih fitur
    df = select_features(df)

    # Menyimpan hasil
    save_dataset(df)

    # Ringkasan
    print("\n" + "=" * 60)
    print("HASIL AKHIR PREPROCESSING")
    print("=" * 60)

    print(f"Total data    : {len(df):,}")
    print(f"Total stasiun : {df['station'].nunique()}")
    print(f"Total fitur   : {len(NUMERIC_FEATURES)}")
    print(
        f"Rentang waktu : "
        f"{df['timestamp'].min()} sampai "
        f"{df['timestamp'].max()}"
    )

    print("\nMissing values akhir:")
    print(df.isnull().sum())

    print("\nContoh data:")
    print(df.head())

    print("\nPREPROCESSING SELESAI!")


if __name__ == "__main__":
    main()
