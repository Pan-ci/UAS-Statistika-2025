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

# -------------------------------
# 1. Load / Buat Dataset
# -------------------------------

# Contoh dataset dummy (GANTI dengan dataset kamu sendiri)
# Kolom: skor_tidur, stres, sosialisasi, label (ya/tidak)
# data = {
#    'skor_tidur': [6, 5, 3, 8, 2, 7, 4, 9, 3, 5, 6, 4, 7, 8, 2],
#    'stres':       [7, 8, 9, 3, 9, 4, 8, 2, 9, 7, 6, 8, 4, 3, 10],
#    'sosialisasi': [2, 3, 1, 7, 1, 5, 2, 8, 1, 3, 2, 1, 6, 7, 1],
#    'label':       ['Ya', 'Ya', 'Ya', 'Tidak', 'Ya', 'Tidak', 'Ya', 'Tidak', 'Ya', 'Ya', 'Tidak', 'Ya', 'Tidak', 'Tidak', 'Ya']
#}

# label encoding memperburuk hasil, ganti menjadi hot encoding
# df = pd.read_csv('Student Mental Health.csv')
# tidak perlu df lagi, sudah impor diatas

df = load_and_clean_data(' -- isi nama file csv nya --')

df['Do you have Depression?'] = df['Do you have Depression?'].map({'Yes': 1, 'No': 0})

df_ohe = pd.get_dummies(df, columns=[
    '', # isi nama kolom "apakah jurusanmu" dari csv
    '', # isi nama kolom "tahun perkuliahan" dari csv
    '', # isi nama kolom "berapa IPK mu?" dari csv
    '', # isi nama kolom jenis kelamin dari csv
    '', # isi nama kolom "apakah kamu sudah menikah?" dari csv
])

# -------------------------------
# 2. Pisahkan Fitur dan Label
# -------------------------------
# X = df[['course_encoded', 'year_encoded', 'cgpa_encoded']]
# y = df_ohe['Do you have Depression?']
# Buang kolom target dari fitur
# Timestamp,Choose your gender,Age,What is your course?,Your current year of Study,What is your CGPA?,Marital status,Do you have Depression?,Do you have Anxiety?,Do you have Panic attack?,Did you seek any specialist for a treatment?
# X = df_ohe.drop(columns=['Mental_Health_Status'])

y = df_ohe['Do you have Depression?']
X = df_ohe.drop(columns=[
    '', # isi nama kolom waktu di csv
    '', # isi nama kolom "apakah kamu mengalami depresi" dari csv
    '', # isi nama kolom "apakah kamu mengalami kecemasan" dari csv
    '', # isi nama kolom "apakah kamu mengalami serangan panik" dari csv
    '' # isi nama kolom "apakah kamu mencari ahli untuk perawatan" dari csv
    ])

print("\nJumlah baris df setelah load_and_clean_data:", df.shape[0])
print("Jumlah baris X:", X.shape[0])
print("Jumlah baris y:", y.shape[0])

'''
# 1. Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Konversi kembali ke DataFrame agar lebih aman
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# 2. SMOTE (pastikan y sudah di-map ke 1/0 sebelumnya!)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled_df, y)
'''

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
knn.fit(X_train, y_train)

# -------------------------------
# 5. Prediksi Data Uji
# -------------------------------
y_pred = knn.predict(X_test)

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
print(classification_report(y_test, y_pred, target_names=['Yes', 'No']))

'''
le_course = LabelEncoder()
le_year = LabelEncoder()
le_cgpa = LabelEncoder()
df['course_encoded'] = le_course.fit_transform(df['What is your course?'])
df['year_encoded'] = le_year.fit_transform(df['Your current year of Study'])
df['cgpa_encoded'] = le_cgpa.fit_transform(df['What is your CGPA?'])

for i, test_point in enumerate(X_test.values):
    test_point_df = pd.DataFrame([test_point], columns=X.columns)
    distances, indices = knn.kneighbors(test_point_df)
    neighbor_labels = y_train.iloc[indices[0]]
    
    # Kembalikan nilai asli fitur (dari encoding)
    course_encoded = int(test_point[0])
    year_encoded = int(test_point[1])
    cgpa_encoded = int(test_point[2])
    
    course_original = le_course.inverse_transform([course_encoded])[0]
    year_original = le_year.inverse_transform([year_encoded])[0]
    cgpa_original = le_cgpa.inverse_transform([cgpa_encoded])[0]
    
    print(f"\nData Uji ke-{i+1}:")
    print("Fitur Asli:")
    print(f"- Course: {course_original}")
    print(f"- Year: {year_original}")
    print(f"- CGPA: {cgpa_original}")
    
    print("Label Tetangga:", list(neighbor_labels.values))
    
    # 2. Voting jumlah kelas
    counts = neighbor_labels.value_counts()
    print("Jumlah masing-masing kelas tetangga:")
    print(counts)
    print("Hasil Voting → Prediksi:", y_pred[i])
'''