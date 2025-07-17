from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from mapping import load_and_clean_data
from imblearn.over_sampling import SMOTE
import joblib

# ==============================
# 1. Load & Preprocessing
# ==============================
df = load_and_clean_data('Student Mental health.csv')

# Simpan data asli sebelum one-hot encoding untuk referensi tampilan
X_original = df.drop(columns=[
    'Timestamp',
    'Do you have Anxiety?',
    'Do you have Panic attack?',
    'Did you seek any specialist for a treatment?'
])

df['Do you have Depression?'] = df['Do you have Depression?'].map({'Yes': 1, 'No': 0})

# One-hot encoding
df_ohe = pd.get_dummies(df, columns=[
    'What is your course?',
    'Your current year of Study',
    'What is your CGPA?',
    'Choose your gender',
    'Marital status',
])

y = df_ohe['Do you have Depression?']

X = df_ohe.drop(columns=[
    'Timestamp',
    'Do you have Depression?',
    'Do you have Anxiety?',
    'Do you have Panic attack?',
    'Did you seek any specialist for a treatment?'
])

# Tambahkan kolom index asli agar bisa ditelusuri kembali setelah SMOTE
X['original_index'] = X.index

# ==============================
# 2. SMOTE
# ==============================
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Simpan kembali index asli
original_indices = X_resampled['original_index']
X_resampled = X_resampled.drop(columns='original_index')

print("\nJumlah baris X_resampled:", X_resampled.shape[0])
print("Jumlah baris y_resampled:", y_resampled.shape[0])

# ==============================
# 3. Split Data
# ==============================
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_resampled, y_resampled, original_indices,
    test_size=0.2, random_state=42, stratify=y_resampled
)

# ==============================
# 4. Train KNN
# ==============================
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# ==============================
# 5. Prediksi & Evaluasi
# ==============================
y_pred = knn.predict(X_test)

joblib.dump(X, 'X_ohe.pkl')
joblib.dump(X_original, 'X_raw.pkl')

print(f"\nK Tetangga Terdekat = {k}")

# Confusion Matrix
print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
print(cm)

# Metrik Evaluasi
print("\n=== Metrik Evaluasi ===")
print("Akurasi :", accuracy_score(y_test, y_pred))
print("Precision (Yes):", precision_score(y_test, y_pred, pos_label=1))
print("Recall (Yes):", recall_score(y_test, y_pred, pos_label=1))
print("F1-Score (Yes):", f1_score(y_test, y_pred, pos_label=1))

# Laporan Klasifikasi
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['No', 'Yes']))

# ==============================
# 6. Tampilkan Data Uji Pertama dan K Tetangga Terdekat
# ==============================

i = 2  # misal data uji ke-3
j = i + 1
test_instance = X_test.iloc[i]
test_index_asli = idx_test.iloc[i]

# Ambil tetangga terdekat dari data uji pertama
distances, indices = knn.kneighbors([test_instance])

# Tampilkan data uji pertama (versi asli)
print("\n=== Data Uji Ke -", j, " (Versi Asli) ===")
print(X_original.loc[test_index_asli])

# Tampilkan data latih terdekat
print("\n=== K Data Latih Terdekat (Versi Asli) ===")
for i, idx in enumerate(indices[0]):
    original_idx = idx_train.iloc[idx]
    print(f"\nTetangga ke-{i+1} (jarak: {distances[0][i]:.4f}):")
    print(X_original.loc[original_idx])
