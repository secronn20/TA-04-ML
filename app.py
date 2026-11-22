
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from flask import Flask, render_template, request
from pathlib import Path
from datetime import datetime

app = Flask(__name__)
app.secret_key = "ta04-lstm-air-quality"

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR / "dataset" / "processed"
    / "air_quality_model_ready.csv.gz"
)
MODEL_FILE = BASE_DIR / "models" / "lstm_pm25.keras"
X_SCALER_FILE = BASE_DIR / "models" / "x_scaler.pkl"
Y_SCALER_FILE = BASE_DIR / "models" / "y_scaler.pkl"
METADATA_FILE = BASE_DIR / "models" / "metadata.json"
METRICS_FILE = BASE_DIR / "outputs" / "evaluation" / "metrics.json"

# =====================================================
# MEMUAT MODEL DAN DATA
# =====================================================

print("Memuat dataset dan model LSTM...")

df = pd.read_csv(DATA_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)

model = tf.keras.models.load_model(
    MODEL_FILE,
    compile=False
)

x_scaler = joblib.load(X_SCALER_FILE)
y_scaler = joblib.load(Y_SCALER_FILE)

with open(METADATA_FILE, "r") as file:
    metadata = json.load(file)

with open(METRICS_FILE, "r") as file:
    metrics = json.load(file)

STATIONS = metadata["stations"]
FEATURES = metadata["numeric_features"]
TIMESTEPS = metadata["timesteps"]

print("Dataset berhasil dimuat.")
print("Model LSTM berhasil dimuat.")
print("Jumlah stasiun:", len(STATIONS))


# =====================================================
# FUNGSI PREDIKSI
# =====================================================

def predict_pm25(station, target_time):
    station_data = df[
        df["station"] == station
    ].sort_values("timestamp").reset_index(drop=True)

    # Ambil 24 jam sebelum waktu target.
    history = station_data[
        station_data["timestamp"] < target_time
    ].tail(TIMESTEPS).copy()

    if len(history) != TIMESTEPS:
        raise ValueError(
            "Data historis tidak mencukupi. "
            "Pilih waktu prediksi yang memiliki 24 jam data sebelumnya."
        )

    # Pastikan seluruh data berjarak satu jam.
    expected_times = pd.date_range(
        end=target_time - pd.Timedelta(hours=1),
        periods=TIMESTEPS,
        freq="h"
    )

    if not history["timestamp"].reset_index(drop=True).equals(
        pd.Series(expected_times)
    ):
        raise ValueError(
            "Data historis tidak berurutan setiap satu jam. "
            "Silakan pilih waktu lain."
        )

    # Normalisasi fitur numerik.
    X_numeric = x_scaler.transform(
        history[FEATURES]
    )

    # One-hot encoding stasiun sesuai metadata training.
    station_encoding = np.zeros(
        (TIMESTEPS, len(STATIONS)),
        dtype=np.float32
    )

    station_index = STATIONS.index(station)
    station_encoding[:, station_index] = 1.0

    # Gabungkan fitur numerik dan identitas stasiun.
    X_input = np.concatenate(
        [X_numeric, station_encoding],
        axis=1
    ).astype(np.float32)

    X_input = np.expand_dims(X_input, axis=0)

    # Prediksi PM2.5.
    prediction_scaled = model.predict(
        X_input,
        verbose=0
    )

    prediction = y_scaler.inverse_transform(
        prediction_scaled
    )[0][0]

    # Cari nilai aktual jika tersedia dalam dataset.
    actual_row = station_data[
        station_data["timestamp"] == target_time
    ]

    actual = None

    if not actual_row.empty:
        actual = float(
            actual_row.iloc[0]["PM2.5"]
        )

    history_values = history["PM2.5"].astype(float).tolist()

    history_labels = [
        timestamp.strftime("%d/%m %H:%M")
        for timestamp in history["timestamp"]
    ]

    return {
        "prediction": float(prediction),
        "actual": actual,
        "history_values": history_values,
        "history_labels": history_labels,
        "target_time": target_time.strftime("%d-%m-%Y %H:%M"),
        "history_start": history["timestamp"].iloc[0].strftime(
            "%d-%m-%Y %H:%M"
        ),
        "history_end": history["timestamp"].iloc[-1].strftime(
            "%d-%m-%Y %H:%M"
        )
    }


# =====================================================
# ROUTE UTAMA
# =====================================================

@app.route("/", methods=["GET", "POST"])
def index():

    prediction_result = None
    error_message = None

    selected_station = STATIONS[0]

    # Default: satu jam setelah data terakhir.
    latest_time = df["timestamp"].max()
    default_target = latest_time + pd.Timedelta(hours=1)

    selected_datetime = default_target.strftime(
        "%Y-%m-%dT%H:%M"
    )

    if request.method == "POST":

        selected_station = request.form.get(
            "station", STATIONS[0]
        )

        selected_datetime = request.form.get(
            "target_time", selected_datetime
        )

        try:
            if selected_station not in STATIONS:
                raise ValueError("Stasiun tidak valid.")

            target_time = pd.Timestamp(
                datetime.strptime(
                    selected_datetime,
                    "%Y-%m-%dT%H:%M"
                )
            )

            result = predict_pm25(
                selected_station,
                target_time
            )

            prediction_result = result

        except ValueError as error:
            error_message = str(error)

        except Exception as error:
            print("Prediction error:", error)
            error_message = (
                "Terjadi kesalahan saat melakukan prediksi. "
                "Periksa kembali data dan model."
            )

    # Statistik dashboard.
    dashboard_stats = {
        "total_data": len(df),
        "total_stations": len(STATIONS),
        "average_pm25": round(df["PM2.5"].mean(), 2),
        "mae": round(metrics["MAE"], 2),
        "rmse": round(metrics["RMSE"], 2),
        "r2": round(metrics["R2"], 4)
    }

    return render_template(
        "index.html",
        stations=STATIONS,
        selected_station=selected_station,
        selected_datetime=selected_datetime,
        prediction=prediction_result,
        error=error_message,
        stats=dashboard_stats
    )


# =====================================================
# MENJALANKAN APLIKASI
# =====================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
