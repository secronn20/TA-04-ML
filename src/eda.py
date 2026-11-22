
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from config import PROCESSED_DATA_DIR, EDA_DIR

# Konfigurasi
INPUT_FILE = PROCESSED_DATA_DIR / "air_quality_cleaned.csv"

FEATURES = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM"
]

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["figure.dpi"] = 120


def load_data():
    print("\n[1] MEMBACA DATASET")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print(f"Total data: {len(df):,}")
    print(f"Jumlah stasiun: {df['station'].nunique()}")

    return df


def plot_pm25_distribution(df):
    print("\n[2] MEMBUAT DISTRIBUSI PM2.5")

    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x="PM2.5",
        bins=60,
        kde=True
    )

    plt.title("Distribusi Konsentrasi PM2.5")
    plt.xlabel("PM2.5 (µg/m³)")
    plt.ylabel("Frekuensi")
    plt.tight_layout()

    output = EDA_DIR / "01_distribusi_pm25.png"
    plt.savefig(output)
    plt.close()

    print(f"Disimpan: {output}")


def plot_pm25_trend(df):
    print("\n[3] MEMBUAT TREN PM2.5")

    daily_data = (
        df.assign(date=df["timestamp"].dt.floor("D"))
        .groupby("date")["PM2.5"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(15, 6))

    plt.plot(
        daily_data["date"],
        daily_data["PM2.5"],
        linewidth=0.8
    )

    plt.title("Tren Rata-rata Harian PM2.5 (2013–2017)")
    plt.xlabel("Tanggal")
    plt.ylabel("Rata-rata PM2.5 (µg/m³)")
    plt.xticks(rotation=30)
    plt.tight_layout()

    output = EDA_DIR / "02_tren_pm25.png"
    plt.savefig(output)
    plt.close()

    print(f"Disimpan: {output}")


def plot_station_comparison(df):
    print("\n[4] MEMBANDINGKAN PM2.5 ANTARSTASIUN")

    station_mean = (
        df.groupby("station")["PM2.5"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(13, 6))

    sns.barplot(
        x=station_mean.index,
        y=station_mean.values
    )

    plt.title("Rata-rata PM2.5 Berdasarkan Stasiun")
    plt.xlabel("Stasiun Pemantauan")
    plt.ylabel("Rata-rata PM2.5 (µg/m³)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output = EDA_DIR / "03_perbandingan_stasiun.png"
    plt.savefig(output)
    plt.close()

    station_mean.to_csv(
        EDA_DIR / "statistik_pm25_per_stasiun.csv",
        header=["mean_PM2.5"]
    )

    print(f"Disimpan: {output}")


def plot_correlation(df):
    print("\n[5] MEMBUAT HEATMAP KORELASI")

    correlation = df[FEATURES].corr()

    plt.figure(figsize=(12, 9))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5
    )

    plt.title("Korelasi Antarfitur Kualitas Udara")
    plt.tight_layout()

    output = EDA_DIR / "04_heatmap_korelasi.png"
    plt.savefig(output)
    plt.close()

    correlation.to_csv(
        EDA_DIR / "matriks_korelasi.csv"
    )

    print(f"Disimpan: {output}")


def plot_pm25_boxplot(df):
    print("\n[6] MEMBUAT BOXPLOT PM2.5")

    plt.figure(figsize=(14, 7))

    sns.boxplot(
        data=df,
        x="station",
        y="PM2.5"
    )

    plt.title("Sebaran PM2.5 pada 12 Stasiun Pemantauan")
    plt.xlabel("Stasiun Pemantauan")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output = EDA_DIR / "05_boxplot_pm25.png"
    plt.savefig(output)
    plt.close()

    print(f"Disimpan: {output}")


def save_descriptive_statistics(df):
    print("\n[7] MENYIMPAN STATISTIK DESKRIPTIF")

    statistics = df[FEATURES].describe().T
    statistics.to_csv(
        EDA_DIR / "statistik_deskriptif.csv"
    )

    print(statistics)
    print("\nStatistik disimpan.")


def main():
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS - BEIJING AIR QUALITY")
    print("=" * 60)

    EDA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    plot_pm25_distribution(df)
    plot_pm25_trend(df)
    plot_station_comparison(df)
    plot_correlation(df)
    plot_pm25_boxplot(df)
    save_descriptive_statistics(df)

    print("\n" + "=" * 60)
    print("EDA SELESAI!")
    print(f"Seluruh hasil tersimpan di: {EDA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
