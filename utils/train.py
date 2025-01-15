from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from utils.model import KGLinkPredictor
from torch import Tensor

def train(train_loader: DataLoader, 
          val_loader: DataLoader, 
          model: KGLinkPredictor, 
          optimizer: Optimizer, 
          device: torch.device, 
          epochs: int) -> Tuple[KGLinkPredictor, int]:
    "" "Train the model for a given number of epochs with train and validation" ""
    
    train_losses = []
    val_losses = []
    
    # Set model to training mode
    model.train()
    
    for epoch in tqdm(range(epochs)):
        
        train_loss = 0
        for batch in train_loader:
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Pass batch through model
            loss = model_pass(batch, model, device)
            
            # Backpropogate and sum loss
            train_loss += loss.item()
            loss.backward()
            
            # Step optimizer
            optimizer.step()
        
        val_loss = 0
        for batch in val_loader:
            
            # Pass batch through model
            with torch.no_grad():
                loss = model_pass(batch, model, device)
            
            # Sum loss
            val_loss += loss.item()
        
        # Compute average loss
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        # Save the best model
        if val_losses[-1] == min(val_losses):
            best_epoch = epoch
            best_model_weights = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model_weights)
    
    return model, best_epoch, plot(train_losses, val_losses)

def pretrain(train_loader: DataLoader, model: KGLinkPredictor, optimizer: Optimizer, device: torch.device) -> KGLinkPredictor:
    "" "Train the model for a given number of epochs with only the training set" ""
        
    # Set model to training mode
    model.train()

    train_loss = 0
    for batch in tqdm(train_loader):
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Pass batch through model
        loss = model_pass(batch, model, device)
        
        # Backpropogate and sum loss
        train_loss += loss.item()
        loss.backward()
        
        # Step optimizer
        optimizer.step()
            
    return model

def model_pass(batch: Tensor, model: KGLinkPredictor, device: torch.device) -> Tensor:
    "" "Pass a batch through the model and return the loss" ""
    
    # Send data to GPU
    batch.to(device)
        
    # Forward pass through model to update node embeddings
    model(batch[:,0], batch[:,1], batch[:,2])
    # Compute loss
    loss = model.loss(batch[:,0], batch[:,1], batch[:,2])
    
    return loss

def plot(train_loss: List[float], val_loss: List[float]) -> plt.Figure:
    "" "Plot the training and validation loss curves" ""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    # Plot loss curves
    ax.plot(train_loss, label='Training Loss', marker='o')
    ax.plot(val_loss, label='Validation Loss', marker='o')
    
    # Add labels and legend
    ax.set_title("Loss Curves")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    ax.legend()
    
    return fig