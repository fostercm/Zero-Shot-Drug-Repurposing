import warnings
warnings.simplefilter(action='ignore')

import pandas as pd
import os
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch
from copy import deepcopy
from torch.nn.init import xavier_uniform_
import numpy as np

def processFiles(path: str) -> pd.DataFrame:
    
    # Load knowledge graph
    print(f"Loading data from path: {path}")
    kg = pd.read_csv(path, header=None, delimiter=r'\t')
    kg = kg.rename(columns={0: 'relation', 1: 'x_type', 2: 'x_name', 3: 'y_type', 4: 'y_name'})
    kg = kg.replace({' ': '_', '-': '_', '/': '_'}, regex=True)
    
    # Ensure knowledge graph is undirected
    reversed_kg = kg.rename(columns={'x_type': 'y_type', 'x_name': 'y_name', 'y_type': 'x_type', 'y_name': 'x_name'})
    kg = pd.concat([kg, reversed_kg], ignore_index=True)
    kg = kg.drop_duplicates()
    
    # Create indices for nodes
    name_to_index_dict = {tuple(row[1]): index for index, row in enumerate(kg[['x_type','x_name']].drop_duplicates().iterrows())}
    kg['x_index'] = kg.apply(lambda row: name_to_index_dict[(row['x_type'], row['x_name'])], axis=1)
    kg['y_index'] = kg.apply(lambda row: name_to_index_dict[(row['y_type'], row['y_name'])], axis=1)
    
    return kg
    
def processNodeData(data: HeteroData, kg: pd.DataFrame, embedding_dim: int) -> None:
        
    # Get unique nodes from knowledge graph
    nodes = deepcopy(kg)
    nodes.drop_duplicates(subset=['x_index'], keep='first', inplace=True)
    
    # Store number of nodes for each node type
    node_dict = {}
    for node_type in nodes['x_type'].unique():
        node_dict[node_type] = len(nodes.loc[nodes['x_type'] == node_type])
        
    # Add nodes to HeteroData object with xavier uniform randomized features
    for node_type in node_dict.keys():
        data[node_type].x = torch.empty(node_dict[node_type], embedding_dim, requires_grad=True)
        xavier_uniform_(data[node_type].x)

def processEdgeData(data: HeteroData, kg: pd.DataFrame) -> None:
    
    edges = deepcopy(kg)
    
    # Create edge dictionary to keep track of within-type indices
    reducedEdges = edges.drop_duplicates(subset=['x_index'], keep='first')
    reducedEdges['group_idx'] = reducedEdges.groupby('x_type').cumcount()
    global_to_local_dict = dict(zip(reducedEdges['x_index'], reducedEdges['group_idx']))

    # Apply edge dictionary
    edges['group_x'] = edges['x_index'].map(global_to_local_dict)
    edges['group_y'] = edges['y_index'].map(global_to_local_dict)
    
    # Group by relation
    groups = edges.groupby('relation')

    for relation, group in groups:
        
        # Get the main group
        subgroups = group.groupby('x_type')
        group_name = list(subgroups.groups)[0]
        group = subgroups.get_group(group_name)
        
        # Get edge indices and create edge list
        x_indices = torch.tensor(group['group_x'].values, dtype=torch.int64)
        y_indices = torch.tensor(group['group_y'].values, dtype=torch.int64)
        edge_list = torch.stack([x_indices, y_indices], dim=0)
            
        # Store in data
        data[group['x_type'].values[0], relation, group['y_type'].values[0]].edge_index = edge_list
    
    # Store edge dict for future use
    data['global_to_local_dict'] = global_to_local_dict

def addBioBERTEmbeddings(data: HeteroData, kg: pd.DataFrame, config: dict) -> None:
    
    # Get disease info
    disease_names = list(pd.read_csv(config['disease_names']))
    disease_embeddings = torch.load(config['disease_embeddings'])
    
    # Modify disease embeddings
    for i,disease in enumerate(disease_names):
        index = kg.loc[kg['x_name'] == disease]['x_index'].values[0] if len(kg.loc[kg['x_name'] == disease]['x_index'].values) > 0 else None
        if index:
            data['disease'].x[data['global_to_local_dict'][index]] = disease_embeddings[i]
    
    # Get drug info
    drug_names = list(pd.read_csv(config['drug_names']))
    drug_embeddings = torch.load(config['drug_embeddings'])

    # Modify drug embeddings
    for i,drug in enumerate(drug_names):
        index = kg.loc[kg['x_name'] == drug]['x_index'].values[0] if len(kg.loc[kg['x_name'] == drug]['x_index'].values) > 0 else None
        if index:
            data['drug'].x[data['global_to_local_dict'][index]] = drug_embeddings[i]
    
def processRelationData(indices: pd.DataFrame) -> pd.DataFrame:
    
    # Group indices by relation
    groups = indices.groupby('relation')
    processed_indices = pd.DataFrame(columns=['x_index','relation','y_index'])

    for relation, group in groups:
        
        # Get the main group
        subgroups = group.groupby('x_type')
        group_name = list(subgroups.groups)[0]
        group = subgroups.get_group(group_name)
        
        # Add only the main group
        processed_indices = pd.concat([processed_indices,group[['x_index','relation','y_index']]])
    
    return processed_indices

def constructDiseaseSimilarity(data: HeteroData, k: int, device: torch.device) -> torch.Tensor:
    
    # Generate and combine the one-hot vectors for all important conditions
    def generateOverallOneHot(disease_idx: int, data: HeteroData) -> torch.Tensor:
        
        # Generate a one-hot vector for a single condition
        def generateOneHot(node_idx: int, edge_type: str, data: HeteroData) -> np.ndarray:
            
            # Get neighbors of the node
            edges = np.array(data[edge_type].edge_index)
            mask = np.where(edges[0] == node_idx)[0]
            neighbors = edges[1][mask]
            
            # Generate the one-hot vector
            one_hot = np.zeros(data[edge_type[2]].num_nodes,dtype=np.int8)
            one_hot[neighbors] = 1
            return one_hot
        
        # Generate the one-hot vectors for the disease with important neighbors and concatenate them
        geneOneHot = generateOneHot(disease_idx,('disease','disease_protein','gene_protein'),data)
        diseaseOneHot = generateOneHot(disease_idx,('disease','disease_disease','disease'),data)
        effectOneHot = generateOneHot(disease_idx,('disease', 'disease_phenotype_negative', 'effect_phenotype'),data) + generateOneHot(disease_idx,('disease', 'disease_phenotype_positive', 'effect_phenotype'),data)
        exposureOneHot = generateOneHot(disease_idx,('disease', 'exposure_disease', 'exposure'),data)
        
        return torch.tensor(np.hstack([geneOneHot,diseaseOneHot,effectOneHot,exposureOneHot]))
    
    # Get the number of diseases and the number of possible neighbors
    num_diseases = data['disease'].num_nodes
    num_possible_neighbors = data['gene_protein'].num_nodes + data['effect_phenotype'].num_nodes + data['exposure'].num_nodes + data['disease'].num_nodes
    
    # Generate the one-hot vectors for all diseases
    oneHots = torch.zeros(num_diseases,num_possible_neighbors)
    for disease_idx in range(num_diseases):
        oneHots[disease_idx] = generateOverallOneHot(disease_idx,data)
    
    # Allocate storage for disease similarities
    disease_similarity_storage = torch.zeros(num_diseases,2,k)
    
    # Transfer one-hots to GPU
    oneHots = oneHots.to(device)
    
    # Calculate the similarity matrix
    similarity_matrix = torch.matmul(oneHots,oneHots.T)
    similarity_matrix.fill_diagonal_(0)
    
    # Take the top k and turn into similarity scores
    values,indices = torch.topk(similarity_matrix,k)
    row_sums = similarity_matrix.sum(dim=1, keepdim=True) + 1e-8
    values = values / row_sums
    
    # Store the similarity values and indices
    disease_similarity_storage[:,0,:] = values
    disease_similarity_storage[:,1,:] = indices
        
    return disease_similarity_storage