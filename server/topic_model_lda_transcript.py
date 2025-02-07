import sqlite3
import numpy as np
from gensim import corpora, models

# Connect to the database
conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# Fetch the tokens from the database
cursor.execute("SELECT rowid, tokens_transcript FROM podcast_episodes WHERE tokens_transcript IS NOT NULL")
rows = cursor.fetchall()

print(f"Loaded {len(rows)} episode transcript tokens.")

# Prepare the corpus from tokens
texts = []
row_ids = []

for row in rows:
    row_id, token_str = row
    tokens = token_str.split()
    texts.append(tokens)
    row_ids.append(row_id)

# Create dictionary and corpus for LDA
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

print(f"Created dictionary and corpus for LDA.")

# Train the LDA model
num_topics = 50
lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

# Get the dominant topic for each episode
dominant_topics = []
for bow in corpus:
    topic_dist = lda_model.get_document_topics(bow)
    if topic_dist:
        dominant_topic = max(topic_dist, key=lambda x: x[1])[0]  # Topic with highest weight
        dominant_topics.append(dominant_topic)
    else:
        dominant_topics.append(None)

# Ensure the 'topic' column exists in the table
cursor.execute("PRAGMA table_info(podcast_episodes);")
columns = [info[1] for info in cursor.fetchall()]
if 'topic_transcript' not in columns:
    print("Adding 'topic_transcript' column to the table.")
    cursor.execute("ALTER TABLE podcast_episodes ADD COLUMN topic_transcript INTEGER")

# Print top words for each topic
for i in range(num_topics):
    words = lda_model.show_topic(i, topn=10)
    top_words = ", ".join([word for word, _ in words])
    print(f"Transcript topic {i}: {top_words}")

# Update the database with the dominant topic
for row_id, topic in zip(row_ids, dominant_topics):
    if topic is not None:
        cursor.execute("UPDATE podcast_episodes SET topic_transcript = ? WHERE rowid = ?", (topic, row_id))

# Commit and close
conn.commit()
conn.close()

print(f"Updated {len(dominant_topics)} episodes with their dominant transcript topics.")