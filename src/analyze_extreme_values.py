
import pandas as pd

from config import PROCESSED_DATA_DIR

INPUT_FILE = PROCESSED_DATA_DIR / "air_quality_cleaned.csv"

FEATURES = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM"
]

def main():
    df = pd.read_csv(INPUT_FILE)

    print("=" * 65)
    print("ANALISIS NILAI EKSTREM DATASET BEIJING AIR QUALITY")
    print("=" * 65)

    print("\n[1] JUMLAH NILAI MAKSIMUM SETIAP FITUR")

    for col in FEATURES:
        maximum = df[col].max()
        count_max = (df[col] == maximum).sum()

        print(
            f"{col:8s} | Maksimum: {maximum:10.2f} "
            f"| Jumlah: {count_max:6,}"
        )

    print("\n[2] PEMERIKSAAN NILAI KANDIDAT KODE KHUSUS")

    candidates = {
        "PM2.5": 999,
        "PM10": 999,
        "SO2": 500,
        "CO": 10000,
    }

    for col, value in candidates.items():
        subset = df[df[col] == value]

        print(f"\n{col} == {value}")
        print(f"Jumlah: {len(subset):,}")

        if not subset.empty:
            print(
                subset[
                    ["timestamp", "station", col]
                ].head(10).to_string(index=False)
            )

    print("\n[3] PERSENTIL DATA")

    print(
        df[FEATURES]
        .quantile([0.90, 0.95, 0.99, 0.999])
        .round(2)
    )

    print("\nANALISIS SELESAI!")

if __name__ == "__main__":
    main()
