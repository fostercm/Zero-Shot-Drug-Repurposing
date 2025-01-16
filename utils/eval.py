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




# our_diseases = ['iEndos']

# indication_predictions  = np.zeros((len(our_diseases),data['drug'].num_nodes))
# contraindication_predictions  = np.zeros((len(our_diseases),data['drug'].num_nodes))

# for i,disease in enumerate(our_diseases):
#     x_index = kg[(kg['x_type']=='disease') & (kg['x_name']==disease)]['x_index'].unique()[0]
    
#     contraindication = torch.ones(data['drug'].num_nodes,dtype=torch.long)*4
#     indication = torch.ones(data['drug'].num_nodes,dtype=torch.long)*5
#     drug = torch.arange(0,data['drug'].num_nodes,dtype=torch.long)
#     query_disease = torch.ones(data['drug'].num_nodes,dtype=torch.long)*x_index
    
#     indication_predictions[i] = torch.sigmoid(model.Decoder(query_disease,indication,drug)).detach().cpu().numpy().flatten()
#     contraindication_predictions[i] = torch.sigmoid(model.Decoder(query_disease,contraindication,drug)).detach().cpu().numpy().flatten()
    
# for i,disease in enumerate(our_diseases):
#     print(disease)
#     print(f'Max Indication: {round(max(indication_predictions[i]),2)} | Threshold: {round(0.5 + indication_threshold,2)}')
#     print(f'Max Contraindication: {round(max(contraindication_predictions[i]),2)} | Threshold: {round(0.5 + contraindication_threshold,2)}\n')

# diseases = ['iEndos']

# ax,fig = plt.subplots(len(diseases),2,figsize=(15,5))
# for i,disease in enumerate(contraindication_predictions):
    
#     fig[0].hist(indication_predictions[i],bins=10)
#     fig[0].set_title(f'{diseases[i]} Indication')
#     fig[0].set_xlabel('Score')
    
#     fig[1].hist(contraindication_predictions[i],bins=10)
#     fig[1].set_title(f'{diseases[i]} Contraindication')
#     fig[1].set_xlabel('Score')
        
# plt.show()

# drugs = kg[kg['x_type']=='drug']['x_name'].unique()
# drugs[torch.topk(torch.Tensor(indication_predictions[0]),10).indices]