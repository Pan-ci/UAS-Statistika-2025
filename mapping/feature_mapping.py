# simulasi jika suatu saar dictionary digunakan oleh banyak program
# atau sering diubah isinya
# untuk kerapihan proyek produksi
# bisa juga menggunakan JSON/YAML

'''
| Kapan Dipisah               | Kapan Disatukan                       |
| --------------------------- | ------------------------------------- |
| Dictionary besar dan banyak | Dictionary kecil dan spesifik         |
| Digunakan di banyak tempat  | Hanya dipakai lokal di modul itu saja |
| Perlu sering diperbarui     | Jarang berubah                        |
| Untuk jaga kerapian proyek  | Untuk kepraktisan saat prototipe      |
'''

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