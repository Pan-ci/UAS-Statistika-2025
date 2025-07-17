from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
# from sklearn.preprocessing import MinMaxScaler
import pandas as pd
# import numpy as np
from mapping import load_and_clean_data
from imblearn.over_sampling import SMOTE
import joblib

# sumber dataset: https://www.kaggle.com/code/shariful07/student-mental-health-data-analysis
df = load_and_clean_data('Student Mental health.csv')

df['Do you have Depression?'] = df['Do you have Depression?'].map({'Yes': 1, 'No': 0})

df_ohe = pd.get_dummies(df, columns=[
    'What is your course?', # isi nama kolom "apakah jurusanmu" dari csv
    'Your current year of Study', # isi nama kolom "tahun perkuliahan" dari csv
    'What is your CGPA?', # isi nama kolom "berapa IPK mu?" dari csv
    'Choose your gender', # isi nama kolom jenis kelamin dari csv
    'Marital status', # isi nama kolom "apakah kamu sudah menikah?" dari csv
])

y = df_ohe['Do you have Depression?']
X = df_ohe.drop(columns=[
    'Timestamp', # isi nama kolom waktu di csv
    'Do you have Depression?', # isi nama kolom "apakah kamu mengalami depresi" dari csv
    'Do you have Anxiety?', # isi nama kolom "apakah kamu mengalami kecemasan" dari csv
    'Do you have Panic attack?', # isi nama kolom "apakah kamu mengalami serangan panik" dari csv
    'Did you seek any specialist for a treatment?' # isi nama kolom "apakah kamu mencari ahli untuk perawatan" dari csv
    ])

# 1. Scaling tidak dibutuhkan disini, mengurangi recall yes model
# 2. SMOTE (pastikan y sudah di-map ke 1/0 sebelumnya!)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nJumlah baris X_resampled:", X_resampled.shape[0])
print("Jumlah baris y_resampled:", y_resampled.shape[0])

# -------------------------------
# 3. Split Data Latih & Uji
# -------------------------------
# test_size=0.3 = 30% data jadi data uji
# kalau data uji terlalu kecil, metrik evaulasi menjadi sangat tinggi
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

# -------------------------------
# 4. Inisialisasi dan Latih KNN
# -------------------------------
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
print("Parameter KNN saat training:", knn.get_params())
knn.fit(X_train, y_train)

# -------------------------------
# 5. Prediksi Data Uji
# -------------------------------
y_pred = knn.predict(X_test)

'''
# Simpan beberapa objek
joblib.dump((knn, X_train.columns.tolist()), 'model_and_columns.pkl')

# Load kembali
model, columns = joblib.load('model_and_columns.pkl')
'''

# Simpan model
joblib.dump(knn, 'knn_model.pkl')

# Simpan kolom one-hot encode agar bisa dipakai ulang saat input manual
joblib.dump(X_train.columns.tolist(), 'feature_columns.pkl')

joblib.dump(X_resampled, 'X_resampled.pkl')
joblib.dump(y_resampled, 'y_resampled.pkl')

# -------------------------------
# 6. Tampilkan Output yang Diminta
# -------------------------------

# 1. Cetak K tetangga terdekat untuk setiap data uji
# distances, indices = knn.kneighbors([test_point])
# Buat DataFrame satu baris dari test_point dan beri nama kolom sesuai X_train
print()
print(f"K Tetangga Terdekat = {k}")

# 3. Confusion Matrix
print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
print(cm)

# 4. Metrik Lain
print("\n=== Metrik Evaluasi ===")
print("Akurasi :", accuracy_score(y_test, y_pred))
print("Precision (Yes):", precision_score(y_test, y_pred, pos_label=1))
print("Recall (Yes):", recall_score(y_test, y_pred, pos_label=1))
print("F1-Score (Yes):", f1_score(y_test, y_pred, pos_label=1))

# 5. Laporan Klasifikasi Lengkap
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['No', 'Yes']))
