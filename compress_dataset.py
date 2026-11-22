
import gzip
import shutil
from pathlib import Path

source = Path(
    "dataset/processed/air_quality_model_ready.csv"
)

destination = Path(
    "dataset/processed/air_quality_model_ready.csv.gz"
)

with source.open("rb") as original:
    with gzip.open(destination, "wb") as compressed:
        shutil.copyfileobj(original, compressed)

print("Kompresi berhasil!")
print("Ukuran asli:", round(source.stat().st_size / 1024 / 1024, 2), "MB")
print("Ukuran gzip:", round(destination.stat().st_size / 1024 / 1024, 2), "MB")