import matplotlib.pyplot as plt
import seaborn as sns
from mapping import load_and_clean_data, map_jurusan
import pandas as pd
import matplotlib.ticker as ticker  # tambahkan di awal file

# sumber dataset: https://www.kaggle.com/code/shariful07/student-mental-health-data-analysis
df = load_and_clean_data('Student Mental health.csv')
df_jurusan = load_and_clean_data('Student Mental health.csv', to_replace=map_jurusan)

tahun_map ={
    'year 1': 'Tahun 1',
    'year 2': 'Tahun 2',
    'year 3': 'Tahun 3',
    'year 4': 'Tahun 4'
}

# 2. Binerisasi target (Depression)
df['Depression'] = df['Do you have Depression?'].map({'Yes': "Ya", 'No': "Tidak"})
df['Depresi'] = df['Do you have Depression?'].map({'Yes': "Ya", 'No': "Tidak"})
df['Do you have Depression?'] = df['Do you have Depression?'].map({'Yes': 1, 'No': 0})
df['Choose your gender'] = df['Choose your gender'].map({'Female': "Perempuan", 'Male': "Laki-laki"})
df['Jenis Kelamin'] = df['Choose your gender']
df['Tahun Perkuliahan'] = df['Your current year of Study'].str.strip().str.lower().map(tahun_map)
df['IPK'] = df['What is your CGPA?'].str.strip().str.lower()
df['Status Pernikahan'] = df['Marital status'].map({'Yes': 'Menikah', 'No': 'Lajang'})
df['Jurusan'] = df_jurusan['What is your course?']
df['Usia'] = df['Age']
# ============================
# 1. Pie chart distribusi target
# ============================
plt.figure(figsize=(5, 5))
counts = df['Depression'].value_counts()

# Pie chart dengan label kategori Ya dan Tidak
counts.plot.pie(
    labels=counts.index,  # Menampilkan "Yes" dan "No" # autopct='%1.1f%%',
    autopct=lambda p: '{:.0f}%'.format(p),
    colors=['#66b3ff', '#ff9999'],
    startangle=90  # (opsional) rotasi agar lebih simetris
)
plt.title('Distribusi Mahasiswa dengan/ tanpa Depresi')
plt.ylabel('')  # Hilangkan label sumbu y
# plt.savefig("Distribusi depresi mahasiswa.png", dpi=300)
# plt.show()

# ============================
# 2. Bar chart per fitur kategorikal
# ============================
def barplot_by_depression(column_name, title, order=None, tahun=False):
    plt.figure(figsize=(8, 4))
    ax = sns.countplot(data=df, x=column_name, hue='Depresi', palette='Set2', order=order)

    # Tambahkan label jumlah di atas bar (tanpa koma)
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge', padding=3)
    
    # Format sumbu Y agar tampil sebagai integer tanpa koma
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))

    plt.title(title)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.ylabel('Jumlah')  # Hilangkan label sumbu y

    # Tambahkan setelah sns.countplot() dan sebelum plt.show()
    max_height = max([bar.get_height() for bar in ax.patches])
    ax.set_ylim(0, max_height * 1.15)  # 15% lebih tinggi dari bar tertinggi
    if tahun is True:
        ax.set_xticklabels([f'{label.get_text()} tahun' for label in ax.get_xticklabels()])

    plt.savefig(f"{title}.png", dpi=300)
    plt.show()

# barplot_by_depression('Jenis Kelamin', 'Jenis Kelamin vs Depresi')
# barplot_by_depression('Tahun Perkuliahan', 'Tahun Perkuliahan vs Depresi')

cgpa_order = ['0 - 1.99', '2.00 - 2.49', '2.50 - 2.99', '3.00 - 3.49', '3.50 - 4.00']
# barplot_by_depression('IPK', 'Indeks Prestasi Kumulatif (IPK) vs Depresi', order=cgpa_order)

marital_order = ['Menikah', 'Lajang']

# barplot_by_depression('Status Pernikahan', 'Status Pernikahan vs Depresi', order=marital_order)

# barplot_by_depression('Usia', 'Usia vs Depresi', tahun=True)

# Threshold minimal frekuensi jurusan untuk dipertahankan
threshold = 3

# Hitung frekuensi tiap jurusan
course_counts = df['Jurusan'].value_counts()

# Jurusan dengan jumlah di bawah threshold → akan jadi 'Other'
minor_courses = course_counts[course_counts < threshold].index

# Ganti jurusan minoritas dengan 'Other'
df['Jurusan'] = df['Jurusan'].replace(minor_courses, 'lain-lain')

# Lihat hasil grouping
print(df['Jurusan'].value_counts())

course_depression = df[df['Depresi'] == 'Ya']['Jurusan'].value_counts()
course_total = df['Jurusan'].value_counts()

course_df = pd.DataFrame({
    'Total': course_total,
    'Depressed': course_depression
}).fillna(0).sort_values(by='Depressed', ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Depressed', y=course_df.index, data=course_df, palette='Reds_r', hue=course_df.index, legend=False)

# Label jumlah di ujung bar
for container in ax.containers:
    ax.bar_label(container, fmt='%d', padding=3, label_type='edge')

ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))

max_width = max([bar.get_width() for bar in ax.patches])
ax.set_xlim(0, max_width * 1.15)

plt.xlabel('Jumlah Mahasiswa dengan Depresi')
plt.title('Jumlah Mahasiswa dengan Depresi per Jurusan (Gabungan Minor → Lain-lain)')
plt.tight_layout()
# plt.savefig("Jumlah Mahasiswa dengan Depresi.png", dpi=300)
# plt.show()

course_ratio = (course_depression / course_total).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=course_ratio.values, y=course_ratio.index, palette='coolwarm', hue=course_ratio.index, legend=False)

# Tambahkan label persentase di ujung bar
for bar in ax.patches:
    width = bar.get_width()
    ax.text(width + 0.01,                   # Sedikit di kanan bar
            bar.get_y() + bar.get_height()/2, 
            f'{width:.0%}',                # Format sebagai persen bulat
            va='center')

# plt.subplots_adjust(right=0.85)  # Memberi ruang di sisi kanan
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0%}'))
max_width = max([bar.get_width() for bar in ax.patches])
ax.set_xlim(0, max_width * 1.075)
plt.xlabel('Rasio Mahasiswa Depresi')
plt.title('Rasio Depresi per Jurusan (Gabungan Minor → Lain-lain)')
plt.tight_layout()
plt.savefig("Rasio Mahasiswa Depresi.png", dpi=300)
plt.show()
'''
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}%'))

def plot_with_label(ax, orient='v', percent=False, show_legend=True):
    """
    Menambahkan label ke batang, mengatur format sumbu, dan menampilkan legend jika diinginkan.
    
    Parameters:
    - ax: objek axes dari plot
    - orient: 'v' untuk vertikal (x-axis kategori), 'h' untuk horizontal
    - percent: True jika nilai rasio/persen
    - show_legend: True untuk menampilkan legend
    """
    # Tambah label pada setiap batang
    if orient == 'v':
        for container in ax.containers:
            ax.bar_label(container, fmt='%d' if not percent else '%.0f%%', padding=3, label_type='edge')
        # Format sumbu Y
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}' if not percent else f'{int(x)}%'))
        max_height = max([bar.get_height() for bar in ax.patches])
        ax.set_ylim(0, max_height * 1.15)
        
    elif orient == 'h':
        if percent:
            for bar in ax.patches:
                width = bar.get_width()
                ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                        f'{width:.0f}%', va='center')
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}%'))
        else:
            for container in ax.containers:
                ax.bar_label(container, fmt='%d', padding=3, label_type='edge')
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))

        max_width = max([bar.get_width() for bar in ax.patches])
        ax.set_xlim(0, max_width * 1.15)
    
    # Tampilkan legend jika diminta
    if show_legend:
        ax.legend(title='Depresi')

# ============================
# 3. Distribusi fitur numerik
# ============================

plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='Do you have Depression?', y='Age', palette='coolwarm', hue='Age', legend=False)
plt.title('Distribusi Usia vs Depresi')
plt.xlabel('Depresi (0 = Tidak, 1 = Ya)')
plt.ylabel('Usia')
plt.tight_layout()
plt.show()

# ============================
# 5. Fitur dengan hanya satu nilai unik
# ============================
single_unique_cols = [col for col in df.columns if df[col].nunique() == 1]
print("Fitur dengan hanya satu nilai unik:", single_unique_cols)

# ============================
# 4. High cardinality features
# ============================
# Misal: Course, kita lihat Top-N
df['Jurusan'] = df_jurusan['What is your course?']
top_courses = df['Jurusan'].value_counts().head(10).index
df_top_courses = df[df['Jurusan'].isin(top_courses)]

plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df_top_courses, y='Jurusan', hue='Depresi', palette='Set1')
plot_with_label(ax, orient='h', show_legend=True)
plt.xlabel('Jumlah')
plt.title('Top 10 Jurusan vs Depresi')
plt.tight_layout()
plt.show()

# ============================
# 6. Korelasi (numerik)
# ============================
# Untuk fitur numerik seperti Age dan target
# df['Depression'] = df['Do you have Depression?'].map({'Yes': 1, 'No': 0})
numerical_cols = ['Age', 'Do you have Depression?']
corr = df[numerical_cols].corr()

plt.figure(figsize=(4, 3))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
'''