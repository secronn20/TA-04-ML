
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from numpy.lib.stride_tricks import sliding_window_view

from config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    EVALUATION_DIR,
    TIMESTEPS,
    EPOCHS,
    BATCH_SIZE
)

# =========================================================
# KONFIGURASI
# =========================================================

INPUT_FILE = PROCESSED_DATA_DIR / "air_quality_model_ready.csv"

MODEL_FILE = MODEL_DIR / "lstm_pm25.keras"
X_SCALER_FILE = MODEL_DIR / "x_scaler.pkl"
Y_SCALER_FILE = MODEL_DIR / "y_scaler.pkl"
METADATA_FILE = MODEL_DIR / "metadata.json"

NUMERIC_FEATURES = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM"
]

TARGET_COLUMN = "PM2.5"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# =========================================================
# 1. MEMBACA DATASET
# =========================================================

def load_dataset():
    print("\n[1] MEMBACA DATASET")

    df = pd.read_csv(INPUT_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(
        ["station", "timestamp"]
    ).reset_index(drop=True)

    stations = sorted(df["station"].unique().tolist())

    print(f"Total data   : {len(df):,}")
    print(f"Jumlah stasiun: {len(stations)}")

    return df, stations


# =========================================================
# 2. PEMBAGIAN DATA DAN NORMALISASI
# =========================================================

def prepare_scalers(df, stations):
    print("\n[2] NORMALISASI DATA")

    # Batas waktu setiap stasiun dihitung terpisah.
    train_rows = []

    for station in stations:
        station_df = df[df["station"] == station]

        n = len(station_df)
        train_end = int(n * TRAIN_RATIO)

        train_rows.append(
            station_df.iloc[:train_end][NUMERIC_FEATURES]
        )

    train_data = pd.concat(train_rows, ignore_index=True)

    # Scaler hanya belajar dari bagian training.
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    x_scaler.fit(train_data[NUMERIC_FEATURES])
    y_scaler.fit(train_data[[TARGET_COLUMN]])

    print("Normalisasi X selesai.")
    print("Normalisasi target PM2.5 selesai.")

    return x_scaler, y_scaler


# =========================================================
# 3. MEMBENTUK SEQUENCE LSTM
# =========================================================

def create_sequences(df, stations, x_scaler, y_scaler):
    print("\n[3] MEMBENTUK SEQUENCE LSTM")

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []

    station_count = len(stations)

    for station_index, station in enumerate(stations):

        station_df = df[
            df["station"] == station
        ].sort_values("timestamp").reset_index(drop=True)

        numeric_values = station_df[NUMERIC_FEATURES].values

        X_numeric = x_scaler.transform(numeric_values)
        y_values = y_scaler.transform(
            station_df[[TARGET_COLUMN]]
        ).flatten()

        # One-hot encoding untuk identitas stasiun.
        station_encoding = np.zeros(
            (len(station_df), station_count),
            dtype=np.float32
        )

        station_encoding[:, station_index] = 1.0

        X_values = np.concatenate(
            [X_numeric, station_encoding],
            axis=1
        ).astype(np.float32)

        n = len(station_df)

        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VALIDATION_RATIO))

        # Setiap window berisi 24 jam sebelumnya.
        windows = sliding_window_view(
            X_values,
            window_shape=TIMESTEPS,
            axis=0
        ).transpose(0, 2, 1)

        # Target mulai dari indeks ke-24.
        train_X = windows[:train_end - TIMESTEPS]
        train_y = y_values[TIMESTEPS:train_end]

        # Validation dan testing tetap memakai konteks
        # 24 jam sebelumnya dari periode yang berurutan.
        val_X = windows[
            train_end - TIMESTEPS:val_end - TIMESTEPS
        ]
        val_y = y_values[train_end:val_end]

        test_X = windows[
            val_end - TIMESTEPS:n - TIMESTEPS
        ]
        test_y = y_values[val_end:]

        X_train_list.append(train_X)
        y_train_list.append(train_y)

        X_val_list.append(val_X)
        y_val_list.append(val_y)

        X_test_list.append(test_X)
        y_test_list.append(test_y)

        print(
            f"{station:20s} | "
            f"Train: {len(train_X):,} | "
            f"Val: {len(val_X):,} | "
            f"Test: {len(test_X):,}"
        )

    X_train = np.concatenate(X_train_list).astype(np.float32)
    y_train = np.concatenate(y_train_list).astype(np.float32)

    X_val = np.concatenate(X_val_list).astype(np.float32)
    y_val = np.concatenate(y_val_list).astype(np.float32)

    X_test = np.concatenate(X_test_list).astype(np.float32)
    y_test = np.concatenate(y_test_list).astype(np.float32)

    print("\nUkuran dataset sequence:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val  :", X_val.shape)
    print("y_val  :", y_val.shape)
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test


# =========================================================
# 4. MEMBANGUN MODEL LSTM
# =========================================================

def build_model(input_shape):
    print("\n[4] MEMBANGUN MODEL LSTM")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.LSTM(
            64,
            return_sequences=True
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.LSTM(32),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(32, activation="relu"),

        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"]
    )

    model.summary()

    return model


# =========================================================
# 5. TRAINING MODEL
# =========================================================

def train_model(model, X_train, y_train, X_val, y_val):
    print("\n[5] TRAINING MODEL LSTM")
    print("Proses training dapat membutuhkan waktu.")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=0.00001
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    model.save(MODEL_FILE)

    print(f"\nModel disimpan: {MODEL_FILE}")

    return history


# =========================================================
# 6. EVALUASI MODEL
# =========================================================

def evaluate_model(model, X_test, y_test, y_scaler):
    print("\n[6] EVALUASI MODEL")

    predictions_scaled = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    actual = y_scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    predicted = y_scaler.inverse_transform(
        predictions_scaled
    ).flatten()

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print("\nHASIL EVALUASI MODEL LSTM")
    print("-" * 45)
    print(f"MAE  : {mae:.4f} µg/m³")
    print(f"RMSE : {rmse:.4f} µg/m³")
    print(f"R²   : {r2:.4f}")
    print("-" * 45)

    metrics = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

    with open(
        EVALUATION_DIR / "metrics.json",
        "w"
    ) as file:
        json.dump(metrics, file, indent=4)

    # Simpan sebagian hasil prediksi untuk pemeriksaan.
    result = pd.DataFrame({
        "PM2.5_Aktual": actual,
        "PM2.5_Prediksi": predicted
    })

    result.to_csv(
        EVALUATION_DIR / "hasil_prediksi.csv",
        index=False
    )

    return actual, predicted


# =========================================================
# 7. VISUALISASI HASIL
# =========================================================

def save_plots(history, actual, predicted):
    print("\n[7] MEMBUAT GRAFIK EVALUASI")

    # Grafik loss
    plt.figure(figsize=(10, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title("Training dan Validation Loss LSTM")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        EVALUATION_DIR / "01_training_loss.png",
        dpi=300
    )

    plt.close()

    # Grafik aktual vs prediksi
    sample_size = min(500, len(actual))

    plt.figure(figsize=(14, 5))

    plt.plot(
        actual[:sample_size],
        label="PM2.5 Aktual"
    )

    plt.plot(
        predicted[:sample_size],
        label="PM2.5 Prediksi"
    )

    plt.title(
        "Perbandingan PM2.5 Aktual dan Prediksi LSTM"
    )

    plt.xlabel("Urutan Data Pengujian")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        EVALUATION_DIR / "02_aktual_vs_prediksi.png",
        dpi=300
    )

    plt.close()

    print("Grafik evaluasi berhasil disimpan.")


# =========================================================
# 8. MENYIMPAN SCALER DAN METADATA
# =========================================================

def save_artifacts(x_scaler, y_scaler, stations):
    joblib.dump(x_scaler, X_SCALER_FILE)
    joblib.dump(y_scaler, Y_SCALER_FILE)

    metadata = {
        "stations": stations,
        "numeric_features": NUMERIC_FEATURES,
        "target": TARGET_COLUMN,
        "timesteps": TIMESTEPS,
        "forecast_horizon": 1
    }

    with open(METADATA_FILE, "w") as file:
        json.dump(metadata, file, indent=4)

    print("\nScaler dan metadata berhasil disimpan.")


# =========================================================
# MAIN PROGRAM
# =========================================================

def main():
    print("=" * 65)
    print("TRAINING MODEL LSTM - BEIJING AIR QUALITY")
    print("=" * 65)

    df, stations = load_dataset()

    x_scaler, y_scaler = prepare_scalers(
        df,
        stations
    )

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = create_sequences(
        df,
        stations,
        x_scaler,
        y_scaler
    )

    model = build_model(
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    )

    history = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val
    )

    actual, predicted = evaluate_model(
        model,
        X_test,
        y_test,
        y_scaler
    )

    save_plots(
        history,
        actual,
        predicted
    )

    save_artifacts(
        x_scaler,
        y_scaler,
        stations
    )

    print("\n" + "=" * 65)
    print("TRAINING DAN EVALUASI SELESAI!")
    print("=" * 65)


if __name__ == "__main__":
    main()
