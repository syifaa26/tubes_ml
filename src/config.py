from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_OPTIONS = {
    "Random Forest": ROOT_DIR / "model_random_forest.pkl",
    "Decision Tree": ROOT_DIR / "model_decision_tree.pkl",
}

SCALER_PATH = ROOT_DIR / "scaler.pkl"

DEFAULT_MODEL_NAME = "Random Forest"
DEFAULT_THRESHOLD = 0.50

# The saved models expose n_features_in_=7 but do not store feature names.
# Keep this order identical to the order used when training the model.
FEATURE_NAMES = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order",
]

NUMERIC_FEATURES = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
]

BINARY_FEATURES = [
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order",
]

FEATURE_LABELS = {
    "distance_from_home": "Jarak dari rumah",
    "distance_from_last_transaction": "Jarak dari transaksi terakhir",
    "ratio_to_median_purchase_price": "Rasio terhadap median pembelian",
    "repeat_retailer": "Retailer pernah digunakan",
    "used_chip": "Menggunakan chip",
    "used_pin_number": "Menggunakan PIN",
    "online_order": "Transaksi online",
}

FEATURE_HELP = {
    "distance_from_home": "Semakin jauh dari pola lokasi normal, semakin berisiko.",
    "distance_from_last_transaction": "Jarak transaksi ini terhadap transaksi sebelumnya.",
    "ratio_to_median_purchase_price": "Nilai pembelian dibanding median historis pengguna.",
    "repeat_retailer": "1 jika merchant/retailer pernah digunakan sebelumnya.",
    "used_chip": "1 jika transaksi memakai chip kartu.",
    "used_pin_number": "1 jika transaksi memakai nomor PIN.",
    "online_order": "1 jika transaksi dilakukan secara online.",
}
