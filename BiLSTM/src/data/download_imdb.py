from __future__ import annotations

from pathlib import Path
from typing import Final

from datasets import load_dataset

DEFAULT_RAW_DIR: Final[str] = "data/raw"
REQUIRED_IMDB_FILES: Final[dict[str, str]] = {
    "train": "train-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
    "unsupervised": "unsupervised-00000-of-00001.parquet",
}
DATASET_NAME: Final[str] = "Kwaai/IMDB_Sentiment"


def raw_imdb_files_exist(raw_dir: str | Path = DEFAULT_RAW_DIR) -> bool:
    """Return whether all required raw IMDb parquet files already exist."""
    raw_path = Path(raw_dir)
    if not raw_path.exists() or not raw_path.is_dir():
        return False
    return all((raw_path / filename).exists() for filename in REQUIRED_IMDB_FILES.values())


def verify_downloaded_files(raw_dir: str | Path = DEFAULT_RAW_DIR) -> None:
    """Raise if any required IMDb parquet files are missing after preparation."""
    raw_path = Path(raw_dir)
    missing_files = [
        filename for filename in REQUIRED_IMDB_FILES.values() if not (raw_path / filename).exists()
    ]
    if missing_files:
        raise FileNotFoundError(f"Missing downloaded IMDb files: {missing_files}")


def download_imdb_dataset(raw_dir: str | Path = DEFAULT_RAW_DIR) -> None:
    """Download the IMDb dataset from Hugging Face and store it as local parquet files."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_NAME)
    dataset["train"].to_pandas().to_parquet(raw_path / REQUIRED_IMDB_FILES["train"], index=False)
    dataset["test"].to_pandas().to_parquet(raw_path / REQUIRED_IMDB_FILES["test"], index=False)
    dataset["unsupervised"].to_pandas().to_parquet(
        raw_path / REQUIRED_IMDB_FILES["unsupervised"],
        index=False,
    )


def ensure_imdb_data_available(raw_dir: str | Path = DEFAULT_RAW_DIR) -> None:
    """Ensure the required raw IMDb parquet files exist locally, downloading if needed."""
    if raw_imdb_files_exist(raw_dir):
        print("Raw IMDb data already available. Skipping download.")
        return

    print("Raw IMDb data not found. Downloading...")
    download_imdb_dataset(raw_dir)
    verify_downloaded_files(raw_dir)
    print("IMDb raw data download complete.")


def main(raw_dir: str = DEFAULT_RAW_DIR) -> None:
    """Run the raw IMDb data preparation entrypoint."""
    ensure_imdb_data_available(raw_dir)


if __name__ == "__main__":
    main()
