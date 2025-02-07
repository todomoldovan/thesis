import sqlite3
import pandas as pd

conn = sqlite3.connect('data.db')

query = "SELECT epDescription, category1, category2, category3, category4, category5, category6, category7, category8, category9, category10 FROM podcast_episodes WHERE epDescription IS NOT NULL"
df = pd.read_sql_query(query, conn)
conn.close()

print(f"Loaded {len(df)} episode descriptions.")

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove <p> at the beginning and </p> at the end (if present)
    text = re.sub(r'^<p>', '', text)  # Remove <p> at the start
    text = re.sub(r'</p>$', '', text)  # Remove </p> at the end
    # Remove non-alphanumeric characters and lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words and len(word) > 2]

# Apply preprocessing
df['tokens'] = df['epDescription'].apply(preprocess_text)

df.to_csv('../data/podcast_episodes_tokenized.csv', index=False)

# text = "banana"
# text_token = preprocess_text(text)
# print(text_token)