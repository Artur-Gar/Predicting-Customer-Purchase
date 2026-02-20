from pathlib import Path

import pandas as pd

from .config import RAW_DATA_DIR


def load_train_test_data(
    train_filename: str = "train_dataset.csv",
    test_filename: str = "test_dataset.csv",
    index_col: int | None = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets from data/raw."""
    train_path = RAW_DATA_DIR / train_filename
    test_path = RAW_DATA_DIR / test_filename

    if not train_path.exists():
        raise FileNotFoundError(f"Missing training file: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test file: {test_path}")

    train_df = pd.read_csv(train_path, index_col=index_col)
    test_df = pd.read_csv(test_path, index_col=index_col)
    return train_df, test_df
