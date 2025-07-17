import pandas as pd
import re

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

map_jurusan = {
    'engine': 'teknik',
    'engin': 'teknik',
    'engineering': 'teknik',

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

    'law': 'hukum',
    'laws': 'hukum',

    'psychology': 'psikologi',

    'pendidikan islam': 'pendidikan islam',
    'islamic education': 'pendidikan islam',

    'human resources': 'humaniora',
    'human sciences': 'humaniora',

    'diploma tesl': 'tesl',
    'taasl': 'tesl',

    'diploma nursing': 'keperawatan',
    'nursing': 'keperawatan',

    'kenms': 'manajemen ekonomi',
    'enm': 'manajemen ekonomi',
    'econs': 'manajemen ekonomi',
}

def load_and_clean_data(filepath, to_replace=None):
    df = pd.read_csv(filepath)
    if to_replace is None:
        to_replace=course_mapping
    df['What is your CGPA?'] = (
        df['What is your CGPA?']
        .str.replace(r'[–—]', '-', regex=True)  # ganti en dan em dash sekaligus
        .str.strip()
    )
    df['course_clean'] = df['What is your course?'].str.lower().str.strip()
    df['course_clean'] = df['course_clean'].replace(to_replace=to_replace)
    df['Your current year of Study'] = df['Your current year of Study'].str.lower().str.strip()

    df.drop(columns=['What is your course?'], inplace=True)
    df.rename(columns={'course_clean': 'What is your course?'}, inplace=True)
    return df

'''
df['What is your CGPA?'] = (
    df['What is your CGPA?']
    .str.replace('–', '-', regex=False)   # en dash
    .str.replace('—', '-', regex=False)   # em dash
    .str.strip()
)
'''