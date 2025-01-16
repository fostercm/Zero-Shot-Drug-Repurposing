import numpy as np
import sys
import json
import torch
from utils.model import KGLinkPredictor
import matplotlib.pyplot as plt

# Read the config file
config_file = sys.argv[1]
with open(config_file) as f:
    config = json.load(f)

# Load data
print('Loading data...')
data_paths = config['data_paths']
data = torch.load(data_paths['data_obj'])
kg = torch.load(data_paths['kg'])
our_diseases = config['test_diseases']
k = config['k']

# Load the model
print('Loading model...')
params = config['model_params']
embedding_dim = data.node_stores[0]['x'].shape[1]
model = KGLinkPredictor(embedding_dim, params['hidden_dim'], data, params['num_heads'], params['num_layers'])
model.load_state_dict(torch.load(config['model_path']))
model.update()

# Allocate storage for predictions
indication_predictions  = np.zeros((len(our_diseases),data['drug'].num_nodes))
contraindication_predictions  = np.zeros((len(our_diseases),data['drug'].num_nodes))
num_drugs = data['drug'].num_nodes

# Get the drug-disease scores
print('Calculating scores...')
for i,disease in enumerate(our_diseases):
    x_index = kg[(kg['x_type']=='disease') & (kg['x_name']==disease)]['x_index'].unique()[0]
    
    contraindication = torch.ones(num_drugs,dtype=torch.long)*2
    indication = torch.ones(num_drugs,dtype=torch.long)*3
    drug = torch.arange(0,num_drugs,dtype=torch.long)
    query_disease = torch.ones(num_drugs,dtype=torch.long)*x_index
    
    indication_predictions[i] = torch.sigmoid(model.Decoder(query_disease,indication,drug)).detach().cpu().numpy().flatten()
    contraindication_predictions[i] = torch.sigmoid(model.Decoder(query_disease,contraindication,drug)).detach().cpu().numpy().flatten()

print('Plotting...')
fig,ax = plt.subplots(len(our_diseases),2,figsize=(15,5))
for i,disease in enumerate(contraindication_predictions):
    
    ax[0].hist(indication_predictions[i],bins=20)
    ax[0].set_title(f'{our_diseases[i]} Indication')
    ax[0].set_xlabel('Score')
    ax[0].set_ylabel('# of drugs')
    
    ax[1].hist(contraindication_predictions[i],bins=20)
    ax[1].set_title(f'{our_diseases[i]} Contraindication')
    ax[1].set_xlabel('Score')
    ax[1].set_ylabel('# of drugs')
fig.savefig(config['plot_path'])

print(f'Finding top {k} drugs...')
drugs = kg[kg['x_type']=='drug']['x_name'].unique()
best_drugs = drugs[torch.topk(torch.Tensor(indication_predictions[0]),k).indices]

print(f'Saving top {k} drugs to {config["output_path"]}...')
with open(config['output_path'], "w") as file:
    for drug in best_drugs:
        file.write(drug + "\n")