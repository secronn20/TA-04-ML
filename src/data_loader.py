
import pandas as pd
from pathlib import Path

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_all_stations():

    csv_files = list(RAW_DATA_DIR.glob("PRSA_Data_*.csv"))

    if len(csv_files) != 12:
        raise ValueError(
            f"Seharusnya terdapat 12 file CSV, "
            f"tetapi ditemukan {len(csv_files)} file."
        )

    all_data = []

    print("=" * 60)
    print("MEMBACA DATASET BEIJING MULTI-SITE AIR QUALITY")
    print("=" * 60)

    for file in sorted(csv_files):

        df = pd.read_csv(file)

        # Mengambil nama stasiun dari nama file
        station = file.name.split("_")[2]

        # Memastikan nama stasiun tersedia
        df["station"] = station

        # Membuat timestamp
        df["timestamp"] = pd.to_datetime(
            df[["year", "month", "day", "hour"]]
        )

        all_data.append(df)

        print(
            f"{station:20s} | "
            f"{len(df):,} baris"
        )

    # Menggabungkan seluruh stasiun
    combined_data = pd.concat(
        all_data,
        ignore_index=True
    )

    # Mengurutkan berdasarkan stasiun dan waktu
    combined_data = combined_data.sort_values(
        ["station", "timestamp"]
    ).reset_index(drop=True)

    # Menyimpan hasil
    output_file = (
        PROCESSED_DATA_DIR /
        "air_quality_12_stations.csv"
    )

    combined_data.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 60)
    print("PENGGABUNGAN DATA BERHASIL")
    print("=" * 60)

    print(f"Total data   : {len(combined_data):,}")
    print(f"Total kolom  : {len(combined_data.columns)}")
    print(
        f"Jumlah stasiun: "
        f"{combined_data['station'].nunique()}"
    )
    print(f"File output  : {output_file}")

    print("\nDaftar stasiun:")
    print(combined_data["station"].unique())

    print("\nContoh data:")
    print(combined_data.head())

    return combined_data


if __name__ == "__main__":
    load_all_stations()
