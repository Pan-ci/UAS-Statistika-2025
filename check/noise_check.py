import pandas as pd

# Contoh: membaca dataset
# df = pd.read_csv("nama_file.csv")  # ganti dengan file dataset kamu
file_path = r"Student Mental Health.csv"
df = pd.read_csv(file_path)

# --- 1. Cek missing values ---
print("Missing values per kolom:")
print(df.isnull().sum())

# --- 2. Cek data duplikat ---
duplikat = df.duplicated().sum()
print(f"\nJumlah baris duplikat: {duplikat}")

# --- 3. Cek outliers secara umum (contoh IQR untuk semua kolom numerik) ---
def deteksi_outliers_iqr(df):
    outlier_summary = {}
    for col in df.select_dtypes(include=['float64', 'int64']):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        outlier_summary[col] = len(outliers)
    return outlier_summary

print("\nOutliers per kolom numerik:")
print(deteksi_outliers_iqr(df))

# --- 4. Cek entri unik untuk kolom kategorikal (mendeteksi ejaan berbeda, noise teks) ---
for col in df.select_dtypes(include='object'):
    print(f"\nNilai unik di kolom '{col}':")
    print(df[col].value_counts())
