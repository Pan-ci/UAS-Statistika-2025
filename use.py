import joblib
import pandas as pd
from collections import Counter

# Load model dan feature columns
knn = joblib.load('knn_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')
print("Parameter KNN yang dimuat:", knn.get_params())

'''

'''
# Load data asli (sebelum OHE) dan data OHE-nya
X_raw = joblib.load('X_raw.pkl')        # Data asli (belum OHE), dalam bentuk DataFrame
X_ohe = joblib.load('X_ohe.pkl')        # Data setelah OHE (digunakan untuk latih KNN)

X_resampled = joblib.load('X_resampled.pkl')
y_resampled = joblib.load('y_resampled.pkl')

# Input manual
manual_input = {
    'What is your course?': 'engineering',
    'Your current year of Study': 'year 3',
    'What is your CGPA?': '3.00 - 3.49',
    'Choose your gender': 'Male',
    'Marital status': 'Yes',
    'Age': 24
}

# Tampilkan input
print("\nInput Pengguna:")
for k, v in manual_input.items():
    print(f"{k}: {v}")

# Ubah ke DataFrame & OHE
manual_df = pd.DataFrame([manual_input])
manual_ohe = pd.get_dummies(manual_df)
manual_ohe = manual_ohe.reindex(columns=feature_columns, fill_value=0)
proba = knn.predict_proba(manual_ohe)
print("Probabilitas klasifikasi (No, Yes):", proba)

# Prediksi
y_pred = knn.predict(manual_ohe)
print("\nKlasifikasi data input:", "Yes" if y_pred[0] == 1 else "No")

# Temukan tetangga terdekat
distances, indices = knn.kneighbors(manual_ohe, n_neighbors=5)

print("Ukuran X_resampled:", len(X_resampled))
print("Index tetangga:", indices)

print("\nInformasi Tetangga Terdekat (berdasarkan data setelah SMOTE):")
for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
    label = y_resampled[idx]
    print(f"Tetangga ke-{i+1}: Index={idx}, Jarak={distance:.4f}, Label Depression={label}")

labels = [y_resampled[idx] for idx in indices[0]]
count = Counter(labels)
mayoritas = count.most_common(1)[0][0]

print("\nMayoritas label tetangga:", 'Yes' if mayoritas == 1 else 'No')
'''

'''
# resampled ganti raw dulu
print("\nInformasi Tetangga Terdekat (berdasarkan data sebelum SMOTE):")
for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
    print(f"\nTetangga ke-{i+1} (Jarak: {distance:.4f}):")
    if idx < len(X_raw):
        label = X_raw.iloc[idx]['Do you have Depression?']
        print(X_raw.iloc[idx].to_string())
        print(f"Label Depression: {label}")
    else:
        print(f"Index {idx} di luar batas data X_raw.")