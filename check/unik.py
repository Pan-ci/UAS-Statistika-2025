import pandas as pd
from mapping import load_and_clean_data

# berdasarkan posisi direktori terminal
# bukan berdasarkan posisi file
df = load_and_clean_data('Student Mental health.csv')
print(df['What is your course?'].value_counts())
