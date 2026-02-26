import re
from pathlib import Path

import pandas as pd
from pandas import DataFrame, Series

_RESOURCES_DIR = Path(__file__).resolve().parent.parent.parent / 'resources'


def load_data(filename: str, target_column: str) -> tuple[DataFrame, Series]:
    """
    Loads a CSV dataset from the resources directory, standardizes column names,
    removes rows with missing target values, and separates features from the target.
    :param filename: CSV filename in the resources directory.
    :param target_column: Name of the target column.
    :return: Tuple of (features DataFrame, target Series).
    """
    file_path = _RESOURCES_DIR / filename
    data = pd.read_csv(file_path)
    # standardize column names
    data = data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))

    # Remove rows with missing target, separate target from predictors
    pruned_data = data.dropna(axis=0, subset=[target_column])
    y = pruned_data[target_column]
    X = pruned_data.drop([target_column], axis=1)
    return X, y
