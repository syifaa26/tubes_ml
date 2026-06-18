from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import DEFAULT_MODEL_NAME, DEFAULT_THRESHOLD, FEATURE_NAMES
from src.model_utils import FALSE_VALUES, TRUE_VALUES, load_model, load_scaler, predict_transactions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluasi tebakan model fraud terhadap data berlabel.")
    parser.add_argument("--input", required=True, help="Path CSV data uji berlabel.")
    parser.add_argument("--target", default="fraud", help="Nama kolom label asli. Default: fraud.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Nama model di config atau path file .pkl.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Ambang fraud.")
    parser.add_argument(
        "--output",
        default="reports/evaluation_predictions.csv",
        help="CSV output berisi label asli, prediksi, dan status benar/salah.",
    )
    return parser.parse_args()


def normalize_target(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUE_VALUES or normalized in {"fraud", "yes"}:
            return 1
        if normalized in FALSE_VALUES or normalized in {"non-fraud", "non_fraud", "safe", "aman"}:
            return 0

    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or numeric not in (0, 1):
        raise ValueError(f"Label target tidak valid: {value!r}")

    return int(numeric)


def error_type(actual: int, predicted: int) -> str:
    if actual == 0 and predicted == 0:
        return "Benar Non-Fraud"
    if actual == 1 and predicted == 1:
        return "Benar Fraud"
    if actual == 0 and predicted == 1:
        return "Salah: Non-Fraud dikira Fraud"
    return "Salah: Fraud dikira Non-Fraud"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    data = pd.read_csv(input_path)
    if args.target not in data.columns:
        raise SystemExit(f"Kolom target '{args.target}' tidak ditemukan di {input_path}.")

    missing_features = [feature for feature in FEATURE_NAMES if feature not in data.columns]
    if missing_features:
        raise SystemExit(f"Kolom fitur belum lengkap: {', '.join(missing_features)}")

    y_true = data[args.target].map(normalize_target)
    model = load_model(args.model)
    scaler = load_scaler()
    prediction_result = predict_transactions(model, data, threshold=args.threshold, scaler=scaler)
    y_pred = prediction_result["prediction"].astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = len(y_true)
    correct = int((y_true == y_pred).sum())
    wrong = int(total - correct)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = None
    if y_true.nunique() == 2:
        roc_auc = roc_auc_score(y_true, prediction_result["fraud_probability"])

    evaluation = data.copy()
    evaluation["actual"] = y_true
    evaluation["actual_label"] = y_true.map({0: "Non-Fraud", 1: "Fraud"})
    evaluation["prediction"] = y_pred
    evaluation["prediction_label"] = prediction_result["prediction_label"]
    evaluation["fraud_probability"] = prediction_result["fraud_probability"]
    evaluation["is_correct"] = y_true == y_pred
    evaluation["error_type"] = [
        error_type(actual, predicted) for actual, predicted in zip(y_true, y_pred)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(output_path, index=False)

    print("Evaluasi Model Fraud")
    print("====================")
    print(f"Model               : {args.model}")
    print(f"Input               : {input_path}")
    print(f"Target              : {args.target}")
    print(f"Threshold           : {args.threshold}")
    print(f"Total data          : {total}")
    print(f"Benar               : {correct}")
    print(f"Salah               : {wrong}")
    print(f"Akurasi             : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision fraud     : {precision:.4f}")
    print(f"Recall fraud        : {recall:.4f}")
    print(f"F1 fraud            : {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC             : {roc_auc:.4f}")
    print()
    print("Confusion Matrix")
    print("----------------")
    print(f"Benar Non-Fraud     : {tn}")
    print(f"Salah Non-Fraud->Fraud : {fp}")
    print(f"Salah Fraud->Non-Fraud : {fn}")
    print(f"Benar Fraud         : {tp}")
    print()
    print("Ringkasan kesalahan")
    print("-------------------")
    print(f"False Positive      : {fp} transaksi aman dikira fraud")
    print(f"False Negative      : {fn} transaksi fraud dikira aman")
    print(f"Detail tersimpan    : {output_path}")


if __name__ == "__main__":
    main()
