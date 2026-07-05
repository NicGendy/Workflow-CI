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
import argparse
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

# Konfigurasi MLflow
if "GITHUB_ACTIONS" in os.environ:
    # Jika berjalan di GitHub CI
    mlflow.set_tracking_uri(f"file:{os.path.join(BASE_DIR, 'mlruns')}")
else:
    # Jika berjalan di laptop/WSL Anda
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("IoT-Network-Intrusion-Detection")


def load_split_data(custom_clean_path=None):
    """Memuat data train dan test hasil dari kriteria 1 secara langsung."""
    # Jika dijalankan di server GitHub Actions
    if custom_clean_path:
        if not os.path.exists(custom_clean_path):
            raise FileNotFoundError(f"Berkas data tidak ditemukan di {custom_clean_path}")
        df = pd.read_csv(custom_clean_path)
        # Memisahkan data latih/uji manual agar fit dan score tetap berjalan normal di server GitHub
        from sklearn.model_selection import train_test_split
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]
        return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # Jika dijalankan secara lokal
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


def main(custom_path=None, n_estimators=100, custom_seed=42):
    # Memuat dataset trafik jaringan IoT
    X_train, X_test, y_train, y_test = load_split_data(custom_path)

    # Mengaktifkan autolog tracking
    mlflow.sklearn.autolog()

    print("[1/3] Menginisialisasi sesi pencatatan MLflow...")
    
    # Menggunakan nested=True agar aman dari tabrakan ID di GitHub Actions
    with mlflow.start_run(run_name="rf_ids_baseline_classifier", nested=True):
        
        # Menggunakan model standar Random Forest dengan parameter dinamis
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=custom_seed,
        )
        
        print("[2/3] Melatih arsitektur Random Forest pada log lalu lintas RT-IoT2022")
        model.fit(X_train, y_train)

        # Evaluasi akurasi metrik data uji
        acc = model.score(X_test, y_test)
        print(f"[3/3] Evaluasi Berhasil. Akurasi Pengujian: {acc:.4f}")
        print("Parameter operasional, visualisasi metrik, dan artefak disimpan di server")



if __name__ == "__main__":
    # Menambahkan argumen baris perintah untuk integrasi MLProject
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    # Jika dipanggil via MLProject, konversi jalur relatif menjadi absolut
    clean_data_path = None
    if args.data_path:
        clean_data_path = os.path.abspath(os.path.join(BASE_DIR, args.data_path))

    # Oper nilai argumen ke dalam fungsi main asli Anda
    main(
        custom_path=clean_data_path, 
        n_estimators=args.n_estimators, 
        custom_seed=args.random_state
    )
