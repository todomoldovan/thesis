import sqlite3
import pandas as pd

import pandas as pd
blm_dic = pd.read_csv('../data/racism_list.csv', header=None)
blm_dic.columns = ['keyword']

# Assume blm_dic is already defined and contains the keyword list
keywords = blm_dic['keyword'].tolist()

# Connect to the database
conn = sqlite3.connect('data.db')
cursor = conn.cursor()

try:
    # Build the dynamic WHERE clause based on the keywords
    where_clause = " OR ".join([f"transcript LIKE '%{keyword}%'" for keyword in keywords])
    
    # Query to filter podcast_episodes by keywords and get unique mp3urls
    query_episodes = f"SELECT DISTINCT mp3url FROM podcast_episodes WHERE {where_clause}"
    cursor.execute(query_episodes)
    filtered_mp3urls = [row[0] for row in cursor.fetchall()]

    # Query to get all mp3urls from speaker_turns
    query_speaker_turns = "SELECT DISTINCT mp3url FROM speaker_turns"
    cursor.execute(query_speaker_turns)
    speaker_turns_mp3urls = {row[0] for row in cursor.fetchall()}

    # Find the intersection of mp3urls
    common_mp3urls = set(filtered_mp3urls) & speaker_turns_mp3urls

    # Print the count of unique matching mp3urls
    print(f"Number of unique matching mp3urls between keyword-filtered episodes and speaker turns: {len(common_mp3urls)}")

except sqlite3.Error as e:
    print(f"An error occurred: {e}")

finally:
    # Close the cursor and connection
    cursor.close()
    conn.close()

print("Got common_mp3urls.")
    
# import sqlite3

# # Assume common_mp3urls is already defined as a set
# conn = sqlite3.connect('data.db')
# cursor = conn.cursor()

# try:
#     # Add a new column "diarized" to the podcast_episodes table (if not exists)
#     # cursor.execute("ALTER TABLE podcast_episodes ADD COLUMN diarized INTEGER DEFAULT 0")
    
#     # Update the "diarized" column for episodes with mp3urls in common_mp3urls
#     for mp3url in common_mp3urls:
#         cursor.execute(
#             "UPDATE podcast_episodes SET diarized = 1 WHERE mp3url = ?",
#             (mp3url,)
#         )
    
#     # Commit the changes
#     conn.commit()
#     print("Diarized column updated successfully.")

# except sqlite3.Error as e:
#     print(f"An error occurred: {e}")

# finally:
#     cursor.close()
#     conn.close()

import sqlite3

# Assume common_mp3urls is already defined as a set
conn = sqlite3.connect('data.db')
cursor = conn.cursor()

try:
    # Add a new column "diarized" to the podcast_episodes table (if not exists)
    #cursor.execute("ALTER TABLE podcast_episodes ADD COLUMN diarized INTEGER DEFAULT 0")
    
    # Chunk the common_mp3urls into batches for more efficient updating
    batch_size = 1000
    common_mp3urls_list = list(common_mp3urls)  # Convert set to list
    
    # Update in batches
    n = 0
    for i in range(0, len(common_mp3urls_list), batch_size):
        batch = common_mp3urls_list[i:i + batch_size]
        
        # Use the "IN" clause to update multiple rows at once
        placeholders = ",".join(["?"] * len(batch))
        cursor.execute(f"""
            UPDATE podcast_episodes 
            SET diarized = 1 
            WHERE mp3url IN ({placeholders})
        """, tuple(batch))
        n = n + 1
        print("Batch {n} complete")

        # Commit after each batch
        conn.commit()

    print("Diarized column updated successfully.")

except sqlite3.Error as e:
    print(f"An error occurred: {e}")

finally:
    cursor.close()
    conn.close()

import sqlite3

# Connect to the database
conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# Add the new column `badDiarization` (if it doesn't exist already)
# cursor.execute('''
#     ALTER TABLE podcast_episodes
#     ADD COLUMN badDiarization INTEGER;
# ''')

# Update the `badDiarization` column based on the condition
cursor.execute('''
    UPDATE podcast_episodes
    SET badDiarization = CASE
        WHEN totalSpLabels = numMainSpeakers THEN 0
        ELSE 1
    END
    WHERE diarized = 1;
''')

# Commit the changes
conn.commit()

# Close the cursor and connection
cursor.close()
conn.close()

print("badDiarization column updated successfully.")
