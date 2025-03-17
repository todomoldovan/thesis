import os
import random
import pandas as pd

def generate_annotation_sets():
    while True:
        try:
            episodes = pd.read_csv('../data/episodes_with_id.csv')
            exclude_categories = ['games', 'christianity', 'politics', 'places']
            filtered_episodes = episodes[~episodes['category1'].isin(exclude_categories)]
            
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

            filtered_episodes = filtered_episodes[filtered_episodes['episode_id'].isin(valid_episode_ids)]
            random_episodes = filtered_episodes.groupby('category1').apply(lambda x: x.sample(5)).reset_index(drop=True)
            
            picked_rows_race = []
            picked_rows_action = []

            def find_episode_file(episode_id):
                for i in range(1, 11):
                    episode_file_path = f'../data/episodes/part_{i}/episode_{episode_id}.csv'
                    if os.path.exists(episode_file_path):
                        return episode_file_path
                return None

            for _, row in random_episodes.iterrows():
                episode_id = row['episode_id']
                episode_file_path = find_episode_file(episode_id)
                
                if episode_file_path and os.path.exists(episode_file_path):
                    episode_df = pd.read_csv(episode_file_path)
                    
                    if 'racialJustice' not in episode_df.columns:
                        continue
                    
                    rows_racialJustice_0 = episode_df[episode_df['racialJustice'] == 0]
                    rows_racialJustice_1 = episode_df[episode_df['racialJustice'] == 1]
                    
                    if not rows_racialJustice_0.empty:
                        random_row_0 = rows_racialJustice_0.sample(1)
                        random_row_0['category1'] = row['category1']
                        picked_rows_race.append(random_row_0)
                    
                    if not rows_racialJustice_1.empty:
                        random_row_1 = rows_racialJustice_1.sample(1)
                        random_row_1['category1'] = row['category1']
                        picked_rows_race.append(random_row_1)

                    if 'collectiveAction' not in episode_df.columns:
                        continue
                    
                    rows_binaryAction_0 = episode_df[episode_df['collectiveAction'] == 0]
                    rows_binaryAction_1 = episode_df[episode_df['collectiveAction'] == 1]
                    
                    if not rows_binaryAction_0.empty:
                        random_row_0_action = rows_binaryAction_0.sample(1)
                        random_row_0_action['category1'] = row['category1']
                        picked_rows_action.append(random_row_0_action)
                    
                    if not rows_binaryAction_1.empty:
                        random_row_1_action = rows_binaryAction_1.sample(1)
                        random_row_1_action['category1'] = row['category1']
                        picked_rows_action.append(random_row_1_action)

            annotation_set_race = pd.concat(picked_rows_race, ignore_index=True)
            annotation_set_action = pd.concat(picked_rows_action, ignore_index=True)
            
            for annotation_set in [annotation_set_race, annotation_set_action]:
                annotation_set['annotator1'] = ''
                annotation_set['annotator2'] = ''
                annotation_set['annotator3'] = ''
            
            annotation_set_race.to_csv('../data/annotation_set_race.csv', index=False)
            annotation_set_action.to_csv('../data/annotation_set_action.csv', index=False)

            num_rows_race = len(annotation_set_race)
            if num_rows_race in [190, 191]:
                return num_rows_race
            
        except Exception as e:
            print(f"Error encountered: {e}. Retrying...")
            continue

num_rows_race = generate_annotation_sets()
print(num_rows_race)
print("Annotation sets saved to '../data/annotation_set_race.csv' and '../data/annotation_set_action.csv'")
