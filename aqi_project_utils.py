from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "Datasets"


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def ensure_existing_file(path_like: str | Path, label: str) -> Path:
    path = resolve_project_path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def ensure_columns(df: pd.DataFrame, required_columns: list[str], source_name: str) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{source_name} is missing required columns: {missing_columns}")


def stack_sequence_groups(
    x_groups: list[np.ndarray], y_groups: list[np.ndarray], context: str
) -> tuple[np.ndarray, np.ndarray]:
    if not x_groups or not y_groups:
        raise ValueError(f"No usable training sequences were created for {context}.")
    return np.vstack(x_groups), np.vstack(y_groups)


def infer_model_features(scaler, fallback_features: list[str]) -> list[str]:
    feature_names = getattr(scaler, "feature_names_in_", None)
    if feature_names is None:
        return list(fallback_features)
    return [str(feature_name) for feature_name in feature_names]


def infer_look_back(model, fallback: int) -> int:
    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list) and input_shape:
        input_shape = input_shape[0]

    if input_shape and len(input_shape) >= 3 and input_shape[1] is not None:
        return int(input_shape[1])

    return fallback


def prepare_recent_feature_frame(
    recent_data: pd.DataFrame, features: list[str], look_back: int, label: str
) -> pd.DataFrame:
    if not isinstance(recent_data, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame.")

    if len(recent_data) < look_back:
        raise ValueError(f"Need at least {look_back} rows in {label}; got {len(recent_data)}.")

    missing_features = [feature for feature in features if feature not in recent_data.columns]
    if missing_features:
        raise ValueError(f"{label} is missing required features: {missing_features}")

    feature_frame = recent_data.tail(look_back).loc[:, features].apply(pd.to_numeric, errors="coerce")
    invalid_columns = [column for column in features if feature_frame[column].isna().any()]
    if invalid_columns:
        raise ValueError(
            f"{label} contains blank or non-numeric values for: {invalid_columns}"
        )

    return feature_frame
