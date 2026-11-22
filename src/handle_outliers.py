
import pandas as pd
import numpy as np

from config import PROCESSED_DATA_DIR

INPUT_FILE = PROCESSED_DATA_DIR / "air_quality_cleaned.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "air_quality_model_ready.csv"

# Nilai maksimum yang dicurigai sebagai batas pembacaan sensor
SUSPECT_VALUES = {
    "PM2.5": [999],
    "PM10": [999],
    "SO2": [500],
    "CO": [10000],
    "O3": [1071],
}

def main():
    print("=" * 65)
    print("PENANGANAN NILAI EKSTREM - BEIJING AIR QUALITY")
    print("=" * 65)

    df = pd.read_csv(INPUT_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(f"\nJumlah data awal: {len(df):,}")

    # 1. Mengubah nilai yang dicurigai menjadi NaN
    print("\n[1] MENANDAI NILAI EKSTREM")

    total_replaced = 0

    for column, values in SUSPECT_VALUES.items():
        mask = df[column].isin(values)
        count = int(mask.sum())

        df.loc[mask, column] = np.nan
        total_replaced += count

        print(f"{column:8s}: {count:5,} nilai ditandai")

    print(f"\nTotal nilai ditandai: {total_replaced:,}")

    # 2. Interpolasi berdasarkan waktu untuk setiap stasiun
    print("\n[2] INTERPOLASI BERDASARKAN WAKTU")

    numeric_columns = list(SUSPECT_VALUES.keys())

    processed_groups = []

    for station, group in df.groupby("station"):
        group = group.sort_values("timestamp").copy()
        group = group.set_index("timestamp")

        group[numeric_columns] = (
            group[numeric_columns]
            .interpolate(method="time", limit_direction="both")
        )

        group = group.reset_index()
        processed_groups.append(group)

        print(f"Stasiun {station:20s} selesai")

    df = pd.concat(processed_groups, ignore_index=True)
    df = df.sort_values(
        ["station", "timestamp"]
    ).reset_index(drop=True)

    # 3. Memeriksa missing values
    print("\n[3] PEMERIKSAAN MISSING VALUES")

    missing = df[numeric_columns].isnull().sum()
    print(missing)

    if missing.sum() > 0:
        raise ValueError(
            "Masih terdapat missing values. "
            "Periksa kembali dataset."
        )

    # 4. Menyimpan dataset
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n[4] DATASET BERHASIL DISIMPAN")
    print(f"Lokasi: {OUTPUT_FILE}")
    print(f"Total data: {len(df):,}")
    print(f"Total stasiun: {df['station'].nunique()}")

    print("\nNilai maksimum setelah penanganan:")
    print(df[numeric_columns].max())

    print("\n" + "=" * 65)
    print("PENANGANAN NILAI EKSTREM SELESAI!")
    print("=" * 65)


if __name__ == "__main__":
    main()
