import pandas as pd

# Contoh: membaca dataset
# df = pd.read_csv("nama_file.csv")  # ganti dengan file dataset kamu
file_path = r"Student Mental Health.csv"
df = pd.read_csv(file_path)
df['Your current year of Study'] = df['Your current year of Study'].str.lower().str.strip()
df['What is your CGPA?'] = df['What is your CGPA?'].str.replace('–', '-', regex=False).str.strip().str.lower()
df['course_clean'] = df['What is your course?'].str.lower().str.strip()

course_mapping = {
    'engine': 'engineering',
    'engin': 'engineering',
    'engineering': 'engineering',

    'koe': 'koe',
    'koe ': 'koe',

    'bcs': 'bcs',
    'bit': 'bit',
    'it': 'bit',

    'benl': 'benl',
    'benl ': 'benl',

    'irkhs': 'kirkhs',
    'kirkhs': 'kirkhs',
    'kirks': 'kirkhs',

    'law': 'laws',
    'laws': 'laws',

    'psychology': 'psychology',

    'pendidikan islam': 'islamic education',
    'islamic education': 'islamic education',

    'human resources': 'human sciences',
    'human sciences': 'human sciences',

    'diploma tesl': 'tesl',
    'taasl': 'tesl',

    'diploma nursing': 'nursing',
    'nursing': 'nursing',

    'kenms': 'economics and management',
    'enm': 'economics and management',
    'econs': 'economics and management',
}

df['course_clean'] = df['course_clean'].replace(course_mapping)

# untuk cek variabel denga tambahan spasi atau karakter tak terlihat
# for val in df['What is your CGPA?'].unique():
#     print(repr(val))

# df['cgpa_clean'] = df['What is your CGPA?'].str.replace('–', '-', regex=False)  # en dash ke dash
# df['cgpa_clean'] = df['cgpa_clean'].str.lower().str.strip()  # hapus spasi dan kecilkan semua

# df.drop(columns=['What is your CGPA?'], inplace=True)
# df.rename(columns={'cgpa_clean': 'What is your CGPA?'}, inplace=True)

df.drop(columns=['What is your course?'], inplace=True)
df.rename(columns={'course_clean': 'What is your course?'}, inplace=True)

# df.drop(columns=['Your current year of Study'], inplace=True)
# df.rename(columns={'year_clean': 'Your current year of Study'}, inplace=True)

# --- 1. Cek missing values ---
print("\nMissing values per kolom:")
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

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df['What is your CGPA?'] = df['What is your CGPA?'].str.replace('–', '-', regex=False).str.strip()
    df['course_clean'] = df['What is your course?'].str.lower().str.strip()
    df['Your current year of Study'] = df['Your current year of Study'].str.lower().str.strip()

    df.drop(columns=['What is your course?'], inplace=True)
    df.rename(columns={'course_clean': 'What is your course?'}, inplace=True)
    return df
