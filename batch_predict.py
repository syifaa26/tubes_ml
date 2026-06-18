from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_MODEL_NAME, DEFAULT_THRESHOLD
from src.model_utils import load_model, load_scaler, predict_transactions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prediksi fraud dari file CSV.")
    parser.add_argument("--input", required=True, help="Path file CSV transaksi.")
    parser.add_argument(
        "--output",
        default="reports/fraud_predictions.csv",
        help="Path output CSV hasil prediksi.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Nama model di config atau path file .pkl.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Ambang probabilitas untuk label Fraud.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(args.model)
    scaler = load_scaler()
    data = pd.read_csv(args.input)
    result = predict_transactions(model, data, threshold=args.threshold, scaler=scaler)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    total = len(result)
    fraud_count = int((result["prediction"] == 1).sum())
    non_fraud_count = total - fraud_count

    print(f"Output tersimpan: {output_path}")
    print(f"Total transaksi : {total}")
    print(f"Fraud           : {fraud_count}")
    print(f"Non-Fraud       : {non_fraud_count}")


if __name__ == "__main__":
    main()
