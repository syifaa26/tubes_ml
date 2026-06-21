import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.model_utils import load_model, load_scaler, predict_transactions

def inject_noise(df, numeric_noise_std=0.10, binary_flip_prob=0.05):
    """
    Menyuntikkan noise ke dataset untuk menguji robustness.
    - numeric_noise_std: standard deviation untuk pengali noise (0.10 = ~10% variasi)
    - binary_flip_prob: probabilitas untuk menukar nilai 0 jadi 1 atau sebaliknya
    """
    noisy_df = df.copy()
    np.random.seed(42) # Untuk reproduksibilitas ilmiah
    
    # 1. Noise pada fitur numerik (mengalikan dengan angka di sekitar 1.0)
    numeric_cols = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']
    for col in numeric_cols:
        noise_multiplier = np.random.normal(1.0, numeric_noise_std, size=len(noisy_df))
        noisy_df[col] = noisy_df[col] * noise_multiplier
        # Pastikan tidak ada nilai negatif karena jarak dan rasio tidak mungkin negatif
        noisy_df[col] = noisy_df[col].clip(lower=0)
        
    # 2. Noise pada fitur biner (menukar nilai dengan probabilitas tertentu)
    binary_cols = ['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']
    for col in binary_cols:
        # Buat mask acak di mana probabilitas True = binary_flip_prob
        flip_mask = np.random.random(size=len(noisy_df)) < binary_flip_prob
        # Lakukan XOR (1^1=0, 0^1=1) untuk menukar nilai
        noisy_df.loc[flip_mask, col] = noisy_df.loc[flip_mask, col].astype(int) ^ 1
        
    return noisy_df

def evaluate(model_name, X, y_true, scaler):
    model = load_model(model_name)
    results = predict_transactions(model, X, scaler=scaler)
    y_pred = results["prediction"]
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"--- Evaluasi {model_name.upper()} ---")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall   : {rec*100:.2f}%")
    print(f"F1 Score : {f1*100:.2f}%")

    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Fraud", "Fraud"])
    disp.plot(cmap=plt.cm.Blues)
    
    clean_name = model_name.replace("model_", "").replace(".pkl", "").replace("_", " ").title()
    plt.title(f"Robustness Test CM: {clean_name}")
    
    # Pastikan folder reports ada
    import os
    os.makedirs("reports", exist_ok=True)
    
    filename = f"reports/cm_robust_{clean_name.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Gambar Confusion Matrix disimpan di: {filename}\n")

def main():
    print("Memuat dataset asli Kaggle...")
    # Menggunakan sampel 100.000 agar komputasi tidak terlalu lama tapi sangat valid secara statistik (p-value kuat)
    df = pd.read_csv('data/kaggle_card_transdata.csv').sample(n=100000, random_state=42)
    
    print("Menyuntikkan noise (gangguan) sebesar 10% ke data numerik dan 5% ke data biner...")
    noisy_df = inject_noise(df, numeric_noise_std=0.10, binary_flip_prob=0.05)
    
    y_true = noisy_df['fraud']
    
    scaler = load_scaler()
    
    print("\nMulai proses pengujian (Robustness Test)...\n")
    evaluate("model_random_forest.pkl", noisy_df, y_true, scaler)
    evaluate("model_decision_tree.pkl", noisy_df, y_true, scaler)
    
    print("Pengujian Robustness Selesai.")

if __name__ == "__main__":
    main()
