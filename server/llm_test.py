import sqlite3
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

import pandas as pd
blm_dic = pd.read_csv('../data/racism_list.csv', header=None)
blm_dic.columns = ['keyword']
keywords = blm_dic['keyword'].tolist()  

# Connect to SQLite database and fetch the relevant data
conn = sqlite3.connect('../data/data.db')
cursor = conn.cursor()

where_clause = " OR ".join([f"LOWER(transcript) LIKE '%{keyword.lower()}%'" for keyword in keywords])

query = f'''
SELECT rowid, turnText
FROM speaker_turns
WHERE mp3url IN 
(SELECT mp3url FROM podcast_episodes WHERE ({where_clause}) AND totalSpLabels = numMainSpeakers)
'''

cursor.execute(query)

# Fetch the turnText values along with the rowid
rows = cursor.fetchall()

# Create a column for storing predictions (if it doesn't exist)
try:
    cursor.execute("ALTER TABLE speaker_turns ADD COLUMN racialJustice INTEGER")
    print("Column 'racialJustice' created.")
except sqlite3.OperationalError:
    print("Column 'racialJustice' already exists.")

# Define the prompt
def generate_test_prompt6(data_point):
    return f"""
            Classify whether the podcast speaker turn is related to racial justice ("1") or not ("0").

            A speaker turn is considered to be related to racial justice if it fits in any of the following descriptions:
            
            *

            *

            *
            
            Return the label "1" or "0" based on the classification.

            text: {data_point}
            label: """.strip()

# Prepare the prompts
texts = [row[1] for row in rows]
texts_prompts = [generate_test_prompt6(text) for text in texts]

# Prepare datasets and load model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

model_dir = "ariannap22/collectiveaction_sft_annotated_only_v6_prompt_v6_p100_synthetic_balanced_more_layered"
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype="float16",
    quantization_config=bnb_config, 
)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Define prediction 
def predict(texts, model, tokenizer):
    y_pred = []
    answers = []
    categories = list(dim_def.keys())

    for i in range(len(texts)):
        prompt = texts[i]
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens=20, 
                        temperature=0.1)
        
        result = pipe(prompt)
        answer = result[0]['generated_text'].split("label:")[-1].strip()
        answers.append(answer)
        
        # Determine the predicted category
        for category in categories:
            if category.lower() in answer.lower():
                y_pred.append(category)
                break
        else:
            y_pred.append("error")
    
    return y_pred, answers

# Get predictions
y_pred, answer = predict(texts_prompts, model, tokenizer)

# Update the database with the predicted class index in the 'racialJustice' column
for row_id, pred in zip([row[0] for row in rows], y_pred):
    cursor.execute('''
    UPDATE speaker_turns
    SET racialJustice = ?
    WHERE rowid = ?
    ''', (pred, row_id))

# Commit the changes to the database
conn.commit()

# Close the connection
conn.close()

print("racialJustice predictions added successfully.")