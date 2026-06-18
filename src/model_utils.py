from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from .config import (
    BINARY_FEATURES,
    DEFAULT_MODEL_NAME,
    DEFAULT_THRESHOLD,
    FEATURE_NAMES,
    MODEL_OPTIONS,
    SCALER_PATH,
)


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded or is incompatible."""


TRUE_VALUES = {"1", "true", "yes", "y", "ya", "iya", "benar"}
FALSE_VALUES = {"0", "false", "no", "n", "tidak", "salah"}


def resolve_model_path(model_name_or_path: str | Path | None = None) -> Path:
    if model_name_or_path is None:
        return MODEL_OPTIONS[DEFAULT_MODEL_NAME]

    value = str(model_name_or_path)
    if value in MODEL_OPTIONS:
        return MODEL_OPTIONS[value]

    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def load_model(model_name_or_path: str | Path | None = None) -> Any:
    model_path = resolve_model_path(model_name_or_path)
    if not model_path.exists():
        raise ModelLoadError(f"File model tidak ditemukan: {model_path}")

    try:
        model = joblib.load(model_path)
    except Exception:
        try:
            with model_path.open("rb") as file:
                model = pickle.load(file)
        except Exception as exc:
            raise ModelLoadError(f"Gagal memuat model {model_path}: {exc}") from exc

    validate_model(model)
    return model


def load_scaler() -> Any:
    if not SCALER_PATH.exists():
        raise ModelLoadError(f"File scaler tidak ditemukan: {SCALER_PATH}")

    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception:
        try:
            with SCALER_PATH.open("rb") as file:
                scaler = pickle.load(file)
        except Exception as exc:
            raise ModelLoadError(f"Gagal memuat scaler {SCALER_PATH}: {exc}") from exc

    return scaler


def validate_model(model: Any) -> None:
    expected = len(FEATURE_NAMES)
    actual = getattr(model, "n_features_in_", expected)
    if actual != expected:
        raise ModelLoadError(
            f"Model membutuhkan {actual} fitur, tetapi konfigurasi berisi {expected} fitur."
        )

    if not hasattr(model, "predict"):
        raise ModelLoadError("Objek model tidak memiliki method predict().")


def prepare_features(records: dict[str, Any] | pd.DataFrame | Iterable[dict[str, Any]]) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        dataframe = records.copy()
    elif isinstance(records, dict):
        dataframe = pd.DataFrame([records])
    else:
        dataframe = pd.DataFrame(list(records))

    missing_columns = [column for column in FEATURE_NAMES if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Kolom wajib belum ada: {', '.join(missing_columns)}")

    features = dataframe.loc[:, FEATURE_NAMES].copy()

    for column in FEATURE_NAMES:
        if column in BINARY_FEATURES:
            features[column] = features[column].map(_normalize_binary)
        else:
            features[column] = pd.to_numeric(features[column], errors="coerce")

    invalid_columns = [column for column in FEATURE_NAMES if features[column].isna().any()]
    if invalid_columns:
        raise ValueError(f"Nilai tidak valid ditemukan pada kolom: {', '.join(invalid_columns)}")

    return features.astype(float)


def predict_transactions(
    model: Any,
    records: dict[str, Any] | pd.DataFrame | Iterable[dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
    scaler: Any = None,
) -> pd.DataFrame:
    features = prepare_features(records)
    
    model_input_features = features.copy()
    if scaler is not None:
        model_input_features[FEATURE_NAMES] = scaler.transform(model_input_features[FEATURE_NAMES].to_numpy())
        
    probabilities = _fraud_probabilities(model, model_input_features)
    predictions = (probabilities >= threshold).astype(int)

    result = features.copy()
    result["fraud_probability"] = np.round(probabilities, 6)
    result["fraud_risk_percent"] = np.round(probabilities * 100, 2)
    result["prediction"] = predictions
    result["prediction_label"] = np.where(predictions == 1, "Fraud", "Non-Fraud")
    return result


def _fraud_probabilities(model: Any, features: pd.DataFrame) -> np.ndarray:
    model_input = _model_input(model, features)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input)
        return probabilities[:, _fraud_class_index(model)]

    predictions = model.predict(model_input)
    return np.array([1.0 if _is_fraud_label(value) else 0.0 for value in predictions])


def _model_input(model: Any, features: pd.DataFrame) -> pd.DataFrame | np.ndarray:
    if hasattr(model, "feature_names_in_"):
        return features
    return features.to_numpy()


def _fraud_class_index(model: Any) -> int:
    classes = np.asarray(getattr(model, "classes_", []))
    if classes.size == 0:
        return -1

    for target in (1, 1.0, "1", "fraud", "Fraud", True):
        matches = np.where(classes == target)[0]
        if matches.size:
            return int(matches[0])

    return int(classes.size - 1)


def _is_fraud_label(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "fraud", "true", "yes"}
    return bool(int(value))


def _normalize_binary(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)

    if value is None or pd.isna(value):
        return np.nan

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUE_VALUES:
            return 1
        if normalized in FALSE_VALUES:
            return 0

    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return np.nan

    if numeric not in (0, 1):
        return np.nan

    return int(numeric)
