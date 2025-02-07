import sqlite3
import json

# Connect to SQLite3 database (or create it)
conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# Create the speaker_turns table
try:
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS speaker_turns (
        mfcc1_sma3Mean REAL,
        mfcc2_sma3Mean REAL,
        mfcc3_sma3Mean REAL,
        mfcc4_sma3Mean REAL,
        F0semitoneFrom27_5Hz_sma3nzMean REAL,
        F1frequency_sma3nzMean REAL,
        turnText TEXT,
        speaker TEXT,
        startTime REAL,
        endTime REAL,
        duration REAL,
        mp3url TEXT,
        turnCount INTEGER,
        inferredSpeakerRole TEXT,
        inferredSpeakerName TEXT
    )
    ''')
    
    # Commit changes
    conn.commit()
    print("Table speaker_turns created successfully.")
    
except sqlite3.Error as e:
    print(f"Error creating table: {e}")
    
finally:
    # Close the connection
    conn.close()

# Open the JSONL file and connect to the database
with open('../data/speakerTurnData.jsonl', 'r') as f:
    # Open a connection to your database
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()

    # Iterate through each line in the JSONL file
    for line in f:
        record = json.loads(line)

        # Display the keys of the record to ensure the fields match (optional for debugging)
        print(record.keys())  # You can comment this line after confirming fields are correct

        # Extract relevant fields and prepare them for insertion
        values_to_insert = (
            record.get('mfcc1_sma3Mean', None),
            record.get('mfcc2_sma3Mean', None),
            record.get('mfcc3_sma3Mean', None),
            record.get('mfcc4_sma3Mean', None),
            record.get('F0semitoneFrom27.5Hz_sma3nzMean', None),
            record.get('F1frequency_sma3nzMean', None),
            record.get('turnText', None),
            ', '.join(record['speaker']) if isinstance(record.get('speaker'), list) else record.get('speaker'),
            record.get('startTime', None),
            record.get('endTime', None),
            record.get('duration', None),
            record.get('mp3url', None),
            record.get('turnCount', None),
            record.get('inferredSpeakerRole', None),
            record.get('inferredSpeakerName', None)
        )

        # Insert the record into the speaker_turns table
        cursor.execute('''
            INSERT INTO speaker_turns (
                mfcc1_sma3Mean, mfcc2_sma3Mean, mfcc3_sma3Mean, mfcc4_sma3Mean, 
                F0semitoneFrom27_5Hz_sma3nzMean, F1frequency_sma3nzMean, turnText, speaker, 
                startTime, endTime, duration, mp3url, turnCount, inferredSpeakerRole, 
                inferredSpeakerName
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', values_to_insert)

    # Commit the changes to the database
    conn.commit()

    # Close the cursor and connection
    cursor.close()
    conn.close()

# Connect to the database
conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# Execute the query to count the number of entries
cursor.execute('SELECT COUNT(*) FROM speaker_turns')
count = cursor.fetchone()[0]

# Print the count
print(f"Number of entries in speaker_turns: {count}")

# Close the cursor and connection
cursor.close()
