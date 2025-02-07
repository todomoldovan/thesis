import sqlite3
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Set up stopwords
stop_words = set(stopwords.words('english'))

# Function to preprocess the text
def preprocess_text(text):
    # Remove <p> at the beginning and </p> at the end (if present)
    text = re.sub(r'^<p>', '', text)  # Remove <p> at the start
    text = re.sub(r'</p>$', '', text)  # Remove </p> at the end
    # Remove non-alphanumeric characters and lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    # Tokenize and remove stopwords
    tokens_transcript = word_tokenize(text)
    return [word for word in tokens_transcript if word not in stop_words and len(word) > 2]

# Connect to the SQLite database
conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# Add 'tokens_transcript' column if it does not exist
try:
    cursor.execute("ALTER TABLE podcast_episodes ADD COLUMN tokens_transcript TEXT")
except sqlite3.OperationalError:
    # If the column already exists, this will be ignored
    pass

# Fetch the relevant data from the database
query = "SELECT rowid, transcript FROM podcast_episodes WHERE transcript IS NOT NULL"
cursor.execute(query)
episodes = cursor.fetchall()

print(f"Loaded {len(episodes)} episode transcripts.")

# Process each episode transcript and update the table
for row in episodes:
    rowid, ep_transcript = row
    tokens_transcript = preprocess_text(ep_transcript)
    tokenized_text = ' '.join(tokens_transcript)  # Convert list of tokens_transcript into a single string
    
    # Update the database with the tokenized text
    update_query = "UPDATE podcast_episodes SET tokens_transcript = ? WHERE rowid = ?"
    cursor.execute(update_query, (tokenized_text, rowid))

# Commit the changes and close the connection
conn.commit()
conn.close()

print(f"Tokenized {len(episodes)} episode transcripts.")