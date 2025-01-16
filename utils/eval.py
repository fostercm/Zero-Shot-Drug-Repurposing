import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.model import KGLinkPredictor
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import seaborn as sns

def predict_link(model: KGLinkPredictor, test_loader: DataLoader, link_type: int, device: torch.device) -> np.ndarray:
    """
        Predicts the links in the test_loader and returns the true and predicted values
    """
    
    true = np.array([])
    pred = np.array([])
    
    for batch in test_loader:
        
        # Isolate the links of the desired type
        batch.to(device)
        links = batch[batch[:,1] == link_type]
        
        # Predict the links we know are true
        true = np.append(true,len(links)*[1])
        predictions = torch.sigmoid(model.Decoder(links[:,0],links[:,1],links[:,2])).detach().cpu().numpy().flatten()
        pred = np.append(pred,predictions)
        
        # Predict the links we know are false
        true = np.append(true,len(links)*[0])
        predictions = torch.sigmoid(model.Decoder(*model.Decoder.random_sample(links[:,0], links[:,1], links[:,2]), loss_tag=False)).detach().cpu().numpy().flatten()
        pred = np.append(pred,predictions)
    
    # Return the predictions
    return np.stack([true,pred],axis=0)

def plot_roc_curves(indication: np.ndarray, contraindication: np.ndarray) -> plt.Figure:
    """
    Plots the roc curve based of the probabilities
    """
    fig, ax = plt.subplots()

    # Compute ROC curve and AUC
    i_fpr, i_tpr, _ = roc_curve(indication[0], indication[1])
    i_score = roc_auc_score(indication[0], indication[1])
    c_fpr, c_tpr, _ = roc_curve(contraindication[0], contraindication[1])
    c_score = roc_auc_score(contraindication[0], contraindication[1])
    
    # Plot the ROC curve
    ax.plot(i_fpr, i_tpr)
    ax.text(0.6, 0.25, f"Indication AUC: {i_score:.2f}", fontsize=10, color='black')
    ax.plot(c_fpr, c_tpr)
    ax.text(0.6, 0.2, f"Contraindication AUC: {c_score:.2f}", fontsize=10, color='black')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(['Indication','Contraindication'])
    
    return fig

def get_best_threshold(true: np.ndarray, pred: np.ndarray) -> float:
    """
    Returns the threshold that maximizes the ROC AUC score
    """
    
    scores = np.array([])
    thresholds = np.arange(0,1.01,0.01)
    
    for threshold in thresholds:
        
        # Modify the predictions based on the threshold
        modified_pred = pred >= threshold
        modified_pred.astype(int)
        
        # Compute the ROC AUC score
        scores = np.append(scores,roc_auc_score(true, modified_pred))

    # Return the threshold with the highest score
    return thresholds[np.argmax(scores)]

def plot_confusion_matrices(indication: np.ndarray, contraindication: np.ndarray) -> plt.Figure:
    """
    Plots the confusion matrices for the indication and contraindication links
    """
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Get the best threshold for each link type
    indication_threshold = get_best_threshold(indication[0], indication[1])
    contraindication_threshold = get_best_threshold(contraindication[0], contraindication[1])
    
    # Compute confusion matrices
    i_matrix = confusion_matrix(indication[0], (indication[1] > indication_threshold).astype(int))
    c_matrix = confusion_matrix(contraindication[0], (contraindication[1] > contraindication_threshold).astype(int))
    
    # Plot the first confusion matrix
    sns.heatmap(i_matrix, annot=True, fmt='d', cmap='hot', ax=ax[0], cbar=False)
    ax[0].set_title('Indication Confusion Matrix')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('Actual')
    
    # Plot the second confusion matrix
    sns.heatmap(c_matrix, annot=True, fmt='d', cmap='hot', ax=ax[1], cbar=False)
    ax[1].set_title('Contraindication Confusion Matrix')
    ax[1].set_xlabel('Predicted')
    ax[1].set_ylabel('Actual')
    
    plt.tight_layout()
    return fig