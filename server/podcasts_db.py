import sqlite3
import json

# Connect to SQLite3 database (or create it)
conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# Create a table based on the column names provided
cursor.execute('''
CREATE TABLE IF NOT EXISTS podcast_episodes (
    transcript TEXT,
    rssUrl TEXT,
    epTitle TEXT,
    epDescription TEXT,
    mp3url TEXT,
    podTitle TEXT,
    lastUpdate REAL,
    itunesAuthor TEXT,
    itunesOwnerName TEXT,
    explicit INTEGER,
    imageUrl TEXT,
    language TEXT,
    createdOn REAL,
    host TEXT,
    podDescription TEXT,
    category1 TEXT,
    category2 TEXT,
    category3 TEXT,
    category4 TEXT,
    category5 TEXT,
    category6 TEXT,
    category7 TEXT,
    category8 TEXT,
    category9 TEXT,
    category10 TEXT,
    oldestEpisodeDate TEXT,
    episodeDateLocalized REAL,
    durationSeconds REAL,
    hostPredictedNames TEXT,
    numUniqueHosts REAL,
    guestPredictedNames TEXT,
    numUniqueGuests REAL,
    neitherPredictedNames TEXT,
    numUniqueNeithers REAL,
    mainEpSpeakers TEXT,
    numMainSpeakers REAL,
    hostSpeakerLabels TEXT,
    guestSpeakerLabels TEXT,
    overlapPropTurnCount REAL,
    avgTurnDuration REAL,
    overlapPropDuration REAL,
    totalSpLabels REAL
)
''')

# Commit the changes
conn.commit()

# Open the JSONL file and connect to the database
with open('../data/episodeLevelData.jsonl', 'r') as f:
    # Open a connection to your database
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()

    # Iterate through each line in the JSONL file
    for line in f:
        record = json.loads(line)

        # Display the keys of the record to ensure the fields match (optional for debugging)
        #print(record.keys())

        # Extract relevant fields and prepare them for insertion
        values_to_insert = (
            record.get('transcript', None),
            record.get('rssUrl', None),
            record.get('epTitle', None),
            record.get('epDescription', None),
            record.get('mp3url', None),
            record.get('podTitle', None),
            record.get('lastUpdate', None),
            record.get('itunesAuthor', None),
            record.get('itunesOwnerName', None),
            record.get('explicit', None),
            record.get('imageUrl', None),
            record.get('language', None),
            record.get('createdOn', None),
            record.get('host', None),
            record.get('podDescription', None),
            record.get('category1', None),
            record.get('category2', None),
            record.get('category3', None),
            record.get('category4', None),
            record.get('category5', None),
            record.get('category6', None),
            record.get('category7', None),
            record.get('category8', None),
            record.get('category9', None),
            record.get('category10', None),
            record.get('oldestEpisodeDate', None),
            record.get('episodeDateLocalized', None),
            record.get('durationSeconds', None),
            record.get('hostPredictedNames', None),
            record.get('numUniqueHosts', None),
            record.get('guestPredictedNames', None),
            record.get('numUniqueGuests', None),
            record.get('neitherPredictedNames', None),
            record.get('numUniqueNeithers', None),
            record.get('mainEpSpeakers', None),
            record.get('numMainSpeakers', None),
            record.get('hostSpeakerLabels', None),
            record.get('guestSpeakerLabels', None),
            record.get('overlapPropTurnCount', None),
            record.get('avgTurnDuration', None),
            record.get('overlapPropDuration', None),
            record.get('totalSpLabels', None)
        )

        # Convert the tuple to a list for modification
        values_to_insert = list(values_to_insert)

        # Modify the values at positions 28, 34, and 36 as needed
        # Position 28: hostPredictedNames (if it's a list, join it as a string)
        if isinstance(values_to_insert[28], list):
            values_to_insert[28] = ', '.join(values_to_insert[28])
        elif isinstance(values_to_insert[28], dict):
            values_to_insert[28] = json.dumps(values_to_insert[28])  # Convert dict to JSON string

        # Position 34: mainEpSpeakers (if it's a list, join it as a string)
        if isinstance(values_to_insert[34], list):
            values_to_insert[34] = ', '.join(values_to_insert[34])
        elif isinstance(values_to_insert[34], dict):
            values_to_insert[34] = json.dumps(values_to_insert[34])  # Convert dict to JSON string

        # Position 36: hostSpeakerLabels (if it's a dictionary, convert to a JSON string)
        if isinstance(values_to_insert[36], dict):
            values_to_insert[36] = json.dumps(values_to_insert[36])  # Convert dict to JSON string

        # Ensure all values are properly handled as strings or None
        for i in range(len(values_to_insert)):
            if values_to_insert[i] is None:
                values_to_insert[i] = 'NULL'  # SQLite's representation for NULL
            elif isinstance(values_to_insert[i], (list, dict)):
                values_to_insert[i] = str(values_to_insert[i])  # Fallback: convert any lists or other types to string

        # Convert the list back to a tuple after modification
        values_to_insert = tuple(values_to_insert)

        # Insert the record into the database
        cursor.execute('''
            INSERT INTO podcast_episodes (
                transcript, rssUrl, epTitle, epDescription, mp3url, podTitle, lastUpdate, itunesAuthor, 
                itunesOwnerName, explicit, imageUrl, language, createdOn, host, podDescription, 
                category1, category2, category3, category4, category5, category6, category7, 
                category8, category9, category10, oldestEpisodeDate, episodeDateLocalized, 
                durationSeconds, hostPredictedNames, numUniqueHosts, guestPredictedNames, 
                numUniqueGuests, neitherPredictedNames, numUniqueNeithers, mainEpSpeakers, 
                numMainSpeakers, hostSpeakerLabels, guestSpeakerLabels, overlapPropTurnCount, 
                avgTurnDuration, overlapPropDuration, totalSpLabels
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', values_to_insert)

    # Commit the changes to the database
    conn.commit()

    # Close the cursor and connection
    cursor.close()