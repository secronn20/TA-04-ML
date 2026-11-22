
from pathlib import Path

# Direktori utama proyek
BASE_DIR = Path(__file__).resolve().parent

# Direktori dataset
DATASET_DIR = BASE_DIR / "dataset"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"

# Direktori model
MODEL_DIR = BASE_DIR / "models"

# Direktori output
OUTPUT_DIR = BASE_DIR / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"

# Parameter forecasting
TARGET_COLUMN = "PM2.5"
TIMESTEPS = 24
FORECAST_HORIZON = 1

# Pembagian dataset
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Parameter LSTM
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Buat folder otomatis jika belum tersedia
for folder in [
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    EDA_DIR,
    EVALUATION_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)
