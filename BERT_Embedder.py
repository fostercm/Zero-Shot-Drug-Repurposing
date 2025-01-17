from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
import json
from tqdm import tqdm

# Read the config file
config_file = sys.argv[1]
with open(config_file) as f:
    config = json.load(f)

# Load the BERT model and tokenizer
print("Loading BERT model...")
model_name = config['BERT_model']
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Read data
print("Reading data...")
data = pd.read_csv(config['feature_table_path'])

# Replace NaNs and aggregate into sentences
data.fillna("", inplace=True)
data["agg"] = data[config['features']].agg(' '.join, axis=1)

# Tokenize each sentence in the 'text' column
print("Tokenizing data...")
tokenized_data = data['agg'].apply(lambda x: tokenizer(x, padding='max_length', truncation=True, max_length=config['max_sentence_length'], return_tensors='pt'))

# Extract input_ids and attention_mask as lists of tensors
input_ids = [item['input_ids'].squeeze(0) for item in tokenized_data]
attention_mask = [item['attention_mask'].squeeze(0) for item in tokenized_data]

# Stack the lists of tensors into single tensors for input to the model
input_ids = torch.stack(input_ids)
attention_mask = torch.stack(attention_mask)

# Ensure inputs are on the same device as the model
device = config['device']
model = model.to(device)

# Create a dataloader
print("Creating dataloader...")
batch_size = config['batch_size']
dataset = TensorDataset(input_ids, attention_mask)
dataloader = DataLoader(dataset, batch_size=batch_size)
sentence_embeddings = []

# Process sentences
print("Processing sentences...")
with torch.no_grad():
    for batch in tqdm(dataloader):
        batch_input_ids, batch_attention_mask = [b.to(device) for b in batch]
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        
        # Get pooled output for each sentence in the batch
        batch_embeddings = outputs.pooler_output
        sentence_embeddings.append(batch_embeddings.cpu())

# Concatenate embeddings back to a single tensor and save
print(f"Saving embeddings to {config['output_path']}...")
sentence_embeddings = torch.cat(sentence_embeddings)
torch.save(sentence_embeddings, config['output_path'])