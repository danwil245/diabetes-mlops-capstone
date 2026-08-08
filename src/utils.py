"""
Shared helpers: dataset schema, loading, and the preprocessing transformer.

Why this file exists:
  train.py, predict.py and app.py all need to agree on (a) the exact feature
  order and (b) how raw values are cleaned before the model sees them. Keeping
  that in one place means the model you TRAIN and the model you SERVE handle
  data identically -- a very common source of bugs in ML systems otherwise.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# The 8 input features, in the order the model expects them. The FastAPI
# request body and any prediction call must use this same order.
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"

# In the Pima dataset a recorded 0 for these columns is impossible in a living
# person (you can't have a glucose or BMI of 0). Those zeros are actually
# "missing value" placeholders, so we convert them to NaN and impute later.
# Pregnancies can legitimately be 0, so it is NOT in this list.
ZERO_INVALID = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Positional indices of ZERO_INVALID columns within FEATURES. We use indices
# (not names) because inside an sklearn Pipeline the data arrives as a plain
# numpy array with no column labels.
_ZERO_INVALID_IDX = [FEATURES.index(c) for c in ZERO_INVALID]


def zero_to_nan(X):
    """Replace impossible 0s with NaN in the invalid-zero columns.

    Defined at module level (not a lambda) so scikit-learn's FunctionTransformer
    can be pickled/joblib-dumped and reloaded later by the API and the tests.
    Works on a DataFrame or a numpy array; always returns a float ndarray.
    """
    arr = np.asarray(X, dtype=float).copy()
    for idx in _ZERO_INVALID_IDX:
        col = arr[:, idx]
        col[col == 0] = np.nan
        arr[:, idx] = col
    return arr


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the diabetes CSV and return a DataFrame with the expected columns."""
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df


def default_data_path() -> str:
    """data/diabetes.csv resolved relative to the project root."""
    return str(Path(__file__).resolve().parents[1] / "data" / "diabetes.csv")
