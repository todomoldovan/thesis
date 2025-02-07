import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('data.db')

# Query the podcast_episodes table
query = "SELECT * FROM speaker_turns"
df = pd.read_sql_query(query, conn)

# Save the dataframe to a CSV file
df.to_csv('../data/speaker_turns.csv', index=False)

# Close the connection
conn.close()
