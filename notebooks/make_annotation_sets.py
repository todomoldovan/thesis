import os
import random
import pandas as pd

episodes = pd.read_csv('../data/episodes_with_id.csv')

exclude_categories = ['games', 'christianity', 'politics', 'places']

filtered_episodes = episodes[~episodes['category1'].isin(exclude_categories)]

# Collect valid episode_ids from the text files
# valid_episode_ids = set()
# for i in range(1, 11):
#     txt_file = f'output_csv_files{i}'
#     if os.path.exists(txt_file):
#         with open(txt_file, 'r') as f:
#             for line in f:
#                 parts = line.strip().split('/')
#                 if parts[-1].startswith('episode_') and parts[-1].endswith('.csv'):
#                     episode_id = int(parts[-1].split('_')[1].split('.')[0])
#                     valid_episode_ids.add(episode_id)
valid_episode_ids = set()
for i in range(1, 11):
    txt_file = f'../data/output_csv_files{i}.txt'
    if os.path.exists(txt_file):
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split('/')
                for part in parts:
                    if part.startswith('episode_') and part.endswith('.csv'):
                        try:
                            episode_id = int(part.replace('episode_', '').replace('.csv', ''))
                            valid_episode_ids.add(episode_id)
                        except ValueError:
                            continue

# Filter episodes based on the valid episode_ids (those files already processed by the racial justice classifier)
filtered_episodes = filtered_episodes[filtered_episodes['episode_id'].isin(valid_episode_ids)]

random_episodes = filtered_episodes.groupby('category1').apply(lambda x: x.sample(5)).reset_index(drop=True)
# for now take however many episodes available in that category
#random_episodes = filtered_episodes.groupby('category1', group_keys=False).apply(lambda x: x.sample(min(len(x), 5))).reset_index(drop=True)

random_episodes[['category1', 'episode_id']]

# Lists to store the picked rows for race and action
picked_rows_race = []
picked_rows_action = []

def find_episode_file(episode_id):
    """Search for an episode file in part_1 to part_10."""
    for i in range(1, 11):
        episode_file_path = f'../data/episodes/part_{i}/episode_{episode_id}.csv'
        if os.path.exists(episode_file_path):
            return episode_file_path
    return None

# Iterate over each episode_id in the random_episodes DataFrame
for _, row in random_episodes.iterrows():
    episode_id = row['episode_id']
    episode_file_path = find_episode_file(episode_id)
    
    if os.path.exists(episode_file_path):
        # Read the episode CSV file
        episode_df = pd.read_csv(episode_file_path)
        
        # # For race-based annotation set (racialJustice = 0 and 1)
        rows_racialJustice_0 = episode_df[episode_df['racialJustice'] == 0]
        rows_racialJustice_1 = episode_df[episode_df['racialJustice'] == 1]
        
        if not rows_racialJustice_0.empty:
            # Randomly pick one row where racialJustice = 0
            random_row_0 = rows_racialJustice_0.sample(1)
            random_row_0['category1'] = row['category1']
            picked_rows_race.append(random_row_0)
        
        if not rows_racialJustice_1.empty:
            # Randomly pick one row where racialJustice = 1
            random_row_1 = rows_racialJustice_1.sample(1)
            random_row_1['category1'] = row['category1']
            picked_rows_race.append(random_row_1)

        # For action-based annotation set (binaryAction = 0 and 1)
        rows_binaryAction_0 = episode_df[episode_df['collectiveAction'] == 0]
        rows_binaryAction_1 = episode_df[episode_df['collectiveAction'] == 1]
        
        if not rows_binaryAction_0.empty:
            # Randomly pick one row where binaryAction = 0
            random_row_0_action = rows_binaryAction_0.sample(1)
            random_row_0_action['category1'] = row['category1']
            picked_rows_action.append(random_row_0_action)
        
        if not rows_binaryAction_1.empty:
            # Randomly pick one row where binaryAction = 1
            random_row_1_action = rows_binaryAction_1.sample(1)
            random_row_1_action['category1'] = row['category1']
            picked_rows_action.append(random_row_1_action)

# Combine all picked rows into a single DataFrame for race and action
annotation_set_race = pd.concat(picked_rows_race, ignore_index=True)
annotation_set_action = pd.concat(picked_rows_action, ignore_index=True)

# # Add blank columns for annotators and classification
for annotation_set in [annotation_set_race, annotation_set_action]:
    annotation_set['annotator1'] = ''
    annotation_set['annotator2'] = ''
    annotation_set['annotator3'] = ''

# Save the final DataFrames to annotation_set_race.csv and annotation_set_action.csv
annotation_set_race.to_csv('../data/annotation_set_race.csv', index=False)
annotation_set_action.to_csv('../data/annotation_set_action.csv', index=False)

print(len(annotation_set_race))
print(len(annotation_set_action))  
print("Annotation sets saved to '../data/annotation_set_race.csv' and '../data/annotation_set_action.csv'")
