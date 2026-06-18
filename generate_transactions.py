from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import pandas as pd

from src.config import FEATURE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate data transaksi untuk uji prediksi massal.")
    parser.add_argument("--rows", type=int, default=500, help="Jumlah transaksi yang dibuat.")
    parser.add_argument("--output", default="data/generated_transactions_500.csv", help="Path output CSV.")
    parser.add_argument("--seed", type=int, default=42, help="Seed random agar hasil bisa diulang.")
    parser.add_argument(
        "--with-label",
        action="store_true",
        help="Tambahkan label fraud sintetis dari aturan independen, bukan dari model.",
    )
    parser.add_argument("--target", default="fraud", help="Nama kolom label jika memakai --with-label.")
    return parser.parse_args()


def make_transaction(is_fraud: bool) -> dict[str, float | int]:
    # Simulasi data produksi yang lebih realistis berdasarkan sifat dataset asli
    if is_fraud:
        # Karakteristik Fraud: Sering terjadi online, pembelian besar, tidak pakai PIN
        distance_from_home = random.choices([random.uniform(0, 20), random.uniform(50, 200)], weights=[0.3, 0.7])[0]
        distance_from_last_transaction = random.choices([random.uniform(0, 5), random.uniform(20, 100)], weights=[0.4, 0.6])[0]
        ratio_to_median_purchase_price = random.uniform(8.0, 25.0)  # Cenderung jauh di atas median
        repeat_retailer = random.choices([0, 1], weights=[0.6, 0.4])[0]
        used_chip = random.choices([0, 1], weights=[0.8, 0.2])[0]
        used_pin_number = random.choices([0, 1], weights=[0.98, 0.02])[0] # Hampir tidak pernah pakai PIN
        online_order = random.choices([0, 1], weights=[0.1, 0.9])[0] # Hampir selalu online
    else:
        # Karakteristik Aman (Non-Fraud): Pembelian wajar, sering pakai PIN/Chip, dekat rumah
        distance_from_home = random.expovariate(1/15.0)  # Rata-rata 15 km
        distance_from_last_transaction = random.expovariate(1/5.0) # Rata-rata 5 km
        ratio_to_median_purchase_price = random.uniform(0.1, 4.0) # Harga belanja wajar
        repeat_retailer = random.choices([0, 1], weights=[0.2, 0.8])[0]
        used_chip = random.choices([0, 1], weights=[0.4, 0.6])[0]
        used_pin_number = random.choices([0, 1], weights=[0.5, 0.5])[0]
        online_order = random.choices([0, 1], weights=[0.6, 0.4])[0]

    return {
        "distance_from_home": round(distance_from_home, 2),
        "distance_from_last_transaction": round(distance_from_last_transaction, 2),
        "ratio_to_median_purchase_price": round(ratio_to_median_purchase_price, 2),
        "repeat_retailer": repeat_retailer,
        "used_chip": used_chip,
        "used_pin_number": used_pin_number,
        "online_order": online_order,
    }


def main() -> None:
    args = parse_args()
    if args.rows <= 0:
        raise SystemExit("--rows harus lebih dari 0.")

    random.seed(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for _ in range(args.rows):
        # Dalam simulasi produksi, rata-rata fraud adalah sebagian kecil saja (misal 8-10%)
        # Untuk tujuan simulasi uji coba ini, kita buat rasio fraud sekitar 15%
        is_fraud = random.random() < 0.15
        
        record = make_transaction(is_fraud)
        if args.with_label:
            record[args.target] = 1 if is_fraud else 0
            
        records.append(record)

    columns = FEATURE_NAMES + ([args.target] if args.with_label else [])
    data = pd.DataFrame(records, columns=columns)
    data.to_csv(output_path, index=False)

    print(f"File dibuat      : {output_path}")
    print(f"Jumlah transaksi : {len(data)}")
    print("Kolom            : " + ", ".join(data.columns))
    if args.with_label:
        fraud_count = int(data[args.target].sum())
        non_fraud_count = int(len(data) - fraud_count)
        print(f"Fraud label      : {fraud_count}")
        print(f"Non-Fraud label  : {non_fraud_count}")


if __name__ == "__main__":
    main()
