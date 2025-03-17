import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

paragraph_turns = pd.read_csv('../data/paragraph_turns.csv')

bert_model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class MyModel(
    nn.Module,
    PyTorchModelHubMixin,
    # optionally, you can add metadata which gets pushed to the model card
    # repo_url="your-repo-url",
    pipeline_tag="text-classification",
    license="mit",
):
    def __init__(self, bert_model, moral_label=2):

        super(MyModel, self).__init__()
        self.bert = bert_model
        bert_dim = 768
        self.invariant_trans = nn.Linear(768, 768)
        self.moral_classification = nn.Sequential(nn.Linear(768,768),
                                                      nn.ReLU(),
                                                      nn.Linear(768, moral_label))

    def forward(self, input_ids, token_type_ids, attention_mask):
        pooled_output = self.bert(input_ids,
                                token_type_ids = token_type_ids,
                                attention_mask = attention_mask).last_hidden_state[:,0,:]


        pooled_output = self.invariant_trans(pooled_output)


        logits = self.moral_classification(pooled_output)

        return logits

def preprocessing(input_text, tokenizer):
    '''
    Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
    '''
    return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                        max_length = 150,
                        padding = 'max_length',
                        return_attention_mask = True,
                        return_token_type_ids = True,  # Add this line
                        return_tensors = 'pt',
                        truncation=True
                   )

import pandas as pd
import torch.nn.functional as F
from joblib import Parallel, delayed
from tqdm import tqdm
import os

# Define MFT values
mft_values = ["care", "harm", "fairness", "cheating", "loyalty", "betrayal",
              "authority", "subversion", "purity", "degradation"]

# Load all models once (avoids reloading in every function call)
models = {mft: MyModel.from_pretrained(f"vjosap/moralBERT-predict-{mft}-in-text", bert_model=bert_model) 
          for mft in mft_values}

# Function to compute MFT scores for a single sentence
def compute_mft_scores(sentence, index, total):
    if isinstance(sentence, list):
        sentence = " ".join(sentence)

    scores = {}
    for mft, model in models.items():
        encodeds = preprocessing(sentence, tokenizer)
        output = model(**encodeds)
        score = F.softmax(output, dim=1)
        scores[mft] = score[0, 1].item()

    # Print progress every 100 rows
    if index % 100 == 0:
        print(f"Processed {index}/{total} sentences ({(index/total)*100:.2f}%)")

    return scores

# Function to save progress periodically
def save_progress(df, filename="../data/paragraph_turns_mft_progress.csv"):
    df.to_csv(filename, index=False)
    print(f"🔹 Saved progress to {filename}")

# Get total number of sentences
total_sentences = len(paragraph_turns)

# Run computation in parallel with threading + progress bar
num_threads = 8  # Adjust based on your system
mft_scores_list = []
batch_size = 500  # Save progress every 500 rows

for i in tqdm(range(0, total_sentences, batch_size), desc="Processing Sentences"):
    batch = paragraph_turns["sentence"].iloc[i:i + batch_size]
    batch_results = Parallel(n_jobs=num_threads, backend="threading")(
        delayed(compute_mft_scores)(sentence, idx, total_sentences) 
        for idx, sentence in enumerate(batch)
    )

    # Append batch results
    mft_scores_list.extend(batch_results)

    # Convert to DataFrame & save progress
    mft_scores_df = pd.DataFrame(mft_scores_list)
    save_progress(mft_scores_df)

# Merge final results with original DataFrame
mft_scores_df = pd.DataFrame(mft_scores_list)
paragraph_turns = pd.concat([paragraph_turns, mft_scores_df], axis=1)

# Final save
save_progress(paragraph_turns, filename="../data/paragraph_turns_mft.csv")
print("Processing complete! Final results saved.")
