import sqlite3
import numpy as np
from cuml.decomposition import LatentDirichletAllocation as cumlLDA
from sklearn.feature_extraction.text import CountVectorizer

# Connect to the database
conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# Fetch the tokens from the database
cursor.execute("SELECT rowid, tokens FROM podcast_episodes WHERE tokens IS NOT NULL")
rows = cursor.fetchall()

print(f"Loaded {len(rows)} episode description tokens.")

# Prepare the corpus from tokens
texts = []
row_ids = []

for row in rows:
    row_id, token_str = row
    # Convert the stored token strings back to lists (assuming tokens were stored as space-separated text)
    tokens = token_str.split()
    texts.append(" ".join(tokens))  # Join back as space-separated string for vectorizer
    row_ids.append(row_id)

# Use CountVectorizer to transform texts into bag-of-words format
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

print("Created bag-of-words matrix for LDA.")

# Train the GPU-based LDA model
num_topics = 50
lda_model = cumlLDA(n_components=num_topics, max_iter=200, random_state=42)
lda_model.fit(X)

print("Trained GPU-based LDA model.")

# Get the dominant topic for each episode
dominant_topics = []
topic_distributions = lda_model.transform(X)
for dist in topic_distributions:
    dominant_topic = np.argmax(dist) if dist.sum() > 0 else None
    dominant_topics.append(dominant_topic)

# Write top words for each topic to a file
feature_names = vectorizer.get_feature_names_out()
with open("description_topics.txt", "w") as file:
    for topic_idx, topic_weights in enumerate(lda_model.components_):
        top_features_idx = topic_weights.argsort()[-10:][::-1]
        top_words = ", ".join([feature_names[i] for i in top_features_idx])
        file.write(f"Topic {topic_idx}: {top_words}\n")
        print(f"Topic {topic_idx}: {top_words}")

# Update the database with the dominant topic
for row_id, topic in zip(row_ids, dominant_topics):
    if topic is not None:
        cursor.execute("UPDATE podcast_episodes SET topic = ? WHERE rowid = ?", (int(topic), row_id))

# Commit and close
conn.commit()
conn.close()

print(f"Updated {len(dominant_topics)} episodes with their dominant topics.")
