"""
Cara menjalankan:
    1. Jalankan MLflow Tracking UI secara lokal di terminal:
         mlflow ui --host 127.0.0.1 --port 5000
    2. Jalankan script ini di terminal lain:
         python modelling.py
    3. Buka http://127.0.0.1:5000 untuk membuka dashboardnya.

Catatan Penting:
    - Sebelum menjalankan script ini, pastikan Anda telah menyalin (copy-paste) 
      folder hasil pembersihan data `rt_iot2022_preprocessing` dari direktori 
      `preprocessing`.
    - Pastikan semua dependensi sudah terinstal di lingkungan Python Anda (Colab, WSL, atau lokal)
      menggunakan perintah: `pip install -r requirements.txt`.
    - Jika server MLflow UI dijalankan dari dalam WSL menggunakan `--host 127.0.0.1`, 
      Anda tetap bisa mengakses dashboard tersebut secara normal melalui browser Windows 
      di alamat http://127.0.0.1:5000 berkat fitur port-forwarding otomatis dari WSL.
"""

import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier

# Konfigurasi Jalur Data & Parameter
BASE_DIR = os.path.dirname(__file__)
TRAIN_PATH = os.path.join(BASE_DIR, "rt_iot2022_preprocessing", "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "rt_iot2022_preprocessing", "test.csv")

TARGET_COL = "Attack_type"
RANDOM_STATE = 42

# Konfigurasi MLflow Lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("IoT-Network-Intrusion-Detection")


def load_split_data():
    """Memuat data train dan test hasil dari kriteria 1 secara langsung."""
    if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            f"Berkas data tidak ditemukan di folder 'rt_iot2022_preprocessing'. "
            f"Pastikan folder tersebut sudah dicopas ke dalam direktori kerja ini."
        )
        
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    
    return X_train, X_test, y_train, y_test


def main():
    # Memuat dataset trafik jaringan IoT
    X_train, X_test, y_train, y_test = load_split_data()

    # Mengaktifkan autolog tracking
    mlflow.sklearn.autolog()

    print("[1/3] Menginisialisasi sesi pencatatan MLflow...")
    # Mengganti run_name menjadi representatif (Model Utama IDS)
    with mlflow.start_run(run_name="rf_ids_baseline_classifier"):
        
        # Menggunakan model standar Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        )
        
        print("[2/3] Melatih arsitektur Random Forest pada log lalu lintas RT-IoT2022...")
        model.fit(X_train, y_train)

        # Evaluasi akurasi metrik data uji
        acc = model.score(X_test, y_test)
        print(f"[3/3] Evaluasi Berhasil. Akurasi Pengujian: {acc:.4f}")
        print("Parameter operasional, visualisasi metrik, dan artefak disimpan di server lokal.")


if __name__ == "__main__":
    main()
