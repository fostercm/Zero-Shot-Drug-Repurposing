import torch
from utils.model import KGLinkPredictor
from utils.train import pretrain, train
from utils.eval import predict_link, plot_roc_curves, plot_confusion_matrices
from itertools import product
from torch.optim import Adam
import os
import sys
import json

# Read the config file
config_file = sys.argv[1]
with open(config_file) as f:
    config = json.load(f)

# Extract config info
data_paths = config['data_paths']
params = config['params']
device = config['device']
epochs = config['epochs']
model_name = config['model_name']

# Load the data
print('Loading data...')
pretrain_loader = torch.load(data_paths['pretrain'])
train_loader = torch.load(data_paths['train'])
val_loader = torch.load(data_paths['val'])
data = torch.load(data_paths['data_obj']).to(device)

# Get embedding dimension
embedding_dim = data.node_stores[0]['x'].shape[1]

# Train each model combination
for hidden_dim, num_heads, num_layers in product(params['hidden_dim'], params['num_heads'], params['num_layers']):
    
    full_model_name = f'{model_name}_D{hidden_dim}_H{num_heads}_L{num_layers}'
    
    if os.path.exists(os.path.join(config['model_path'],full_model_name)):
        print(f'\nModel {full_model_name} already exists. Skipping...')
        continue
    
    print(f'\nTraining model: {full_model_name}')
    
    # Initialize model and optimizer
    model = KGLinkPredictor(embedding_dim, hidden_dim, data, num_heads, num_layers).to(device)
    optimizer = Adam(model.parameters(), lr=params['learning_rate'])
    
    # Pretrain model
    print('Pretraining model...')
    model = pretrain(pretrain_loader, model, optimizer, device)
    
    # Train model
    print('Finetuning model...')
    model, _, training_fig = train(train_loader, val_loader, model, optimizer, device, epochs)
    
    # Save model
    model_path = os.path.join(config['model_path'],full_model_name+'.pt')
    torch.save(model.state_dict(), model_path)
    
    # Save training plot
    plot_path = os.path.join(config['plot_path'],full_model_name+'_')
    training_fig.savefig(plot_path+'training.png')
    
    # Evaluate model and save evaluation plots
    contraindication = predict_link(model,val_loader,2,device)
    indication = predict_link(model,val_loader,3,device)
    
    confusion_fig = plot_confusion_matrices(indication, contraindication)
    confusion_fig.savefig(plot_path+'confusion.png')
    
    roc_fig = plot_roc_curves(indication, contraindication)
    roc_fig.savefig(plot_path+'roc.png')