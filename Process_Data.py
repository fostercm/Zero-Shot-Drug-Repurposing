import warnings
warnings.simplefilter(action='ignore')

import pandas as pd
import os
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch
from utils.data import processFiles, processNodeData, processEdgeData, processRelationData, addBioBERTEmbeddings, constructDiseaseSimilarity
import sys
import json

# Read the config file
config_file = sys.argv[1]
with open(config_file) as f:
    config = json.load(f)

# Get data path and process knowledge graph
kg = processFiles(config['graph_file'])
data = HeteroData()
device = config['device']
batch_size = config['batch_size']
k = config['k']
output_paths = config['output_paths']

# Process node data
print("Processing node data...")
processNodeData(data, kg, config['embedding_dim'])
if config['BERT_files']:
    addBioBERTEmbeddings(data, kg, config['BERT_files'])

# Process edge data
print("Processing edge data...")
processEdgeData(data, kg)    

# Get dictionary of names to numbers for relations
print("Setting up dataloaders...")
name_to_num = dict(zip(kg['relation'], pd.factorize(kg['relation'])[0]))

# Get pretrain indices and process relation
raw_pretrain_indices = kg[['x_index','x_type','relation','y_index']]
raw_pretrain_indices['relation'] = raw_pretrain_indices['relation'].map(name_to_num)
pretrain_indices = processRelationData(raw_pretrain_indices)

# Get finetuneindices and process relation
raw_finetune_indices = kg[['x_index','x_type','relation','y_index']]
raw_finetune_indices = raw_finetune_indices.loc[(raw_finetune_indices['relation'] == 'contraindication') | 
                                                (raw_finetune_indices['relation'] == 'indication') | 
                                                (raw_finetune_indices['relation'] == 'off_label_use')]
raw_finetune_indices['relation'] = raw_finetune_indices['relation'].map(name_to_num)
finetune_indices = processRelationData(raw_finetune_indices)


# Make sure indices are a compatible datatype
pretrain_indices = pretrain_indices.astype({'x_index': 'int64', 'relation': 'int64', 'y_index': 'int64'})
finetune_indices = finetune_indices.astype({'x_index': 'int64', 'relation': 'int64', 'y_index': 'int64'})

# Turn into tensors
pdata = torch.tensor(pretrain_indices.values,dtype=torch.int32).to(device)
fdata = torch.tensor(finetune_indices.values,dtype=torch.int32).to(device)

# Split finetuning data into train, validation, and test sets
train_data, val_data, test_data = torch.utils.data.random_split(fdata, [0.8,0.1,0.1])

# Combine the second and third splits
nontraining_data = torch.vstack([fdata[val_data.indices], fdata[test_data.indices]])
pdata_set = set(map(tuple, pdata.tolist()))
combined_set = set(map(tuple, nontraining_data.tolist()))
filtered_set = pdata_set - combined_set

# Convert back to tensor
pretrain_data = torch.tensor(list(filtered_set))

# Create dataloaders
pretrain_loader = DataLoader(pretrain_data, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Data must be undirected for proper GNN Message Passing
data = T.ToUndirected()(data)

print("Processing disease similarities...")
data['similarity'] = constructDiseaseSimilarity(data,k,device)

print("Saving data...")
torch.save(data, output_paths['data_obj'])
torch.save(pretrain_loader, output_paths['pretrain'])
torch.save(train_loader, output_paths['train'])
torch.save(val_loader, output_paths['val'])
torch.save(test_loader, output_paths['test'])
torch.save(kg, output_paths['graph'])

print("Data processing complete!")