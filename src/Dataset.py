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

def processFiles(path, fileName):
        
    # Get data path and import KG file with column names
    dataPath = path + r'/data/' + fileName
    print(f"Loading data from path: {dataPath}")
    kg = pd.read_csv(dataPath, header=None, delimiter=r'\t')
    kg = kg.rename(columns={0: 'relation', 1: 'x_type', 2: 'x_name', 3: 'y_type', 4: 'y_name'})
    kg['x_type']= kg['x_type'].apply(lambda x: x.replace("/","_"))
    kg['y_type']= kg['y_type'].apply(lambda x: x.replace("/","_"))
    kg['relation']= kg['relation'].apply(lambda x: x.replace("-","_"))
    kg['relation']= kg['relation'].apply(lambda x: x.replace(" ","_"))
    kg['x_name']= kg['x_name'].apply(lambda x: x.replace("-","_"))
    kg['y_name']= kg['y_name'].apply(lambda x: x.replace("-","_"))
    
    name_to_index_dict = {tuple(row[1]): index for index, row in enumerate(kg[['x_type','x_name']].drop_duplicates().iterrows())}
    kg['x_index'] = kg.apply(lambda row: name_to_index_dict[(row['x_type'], row['x_name'])], axis=1)
    kg['y_index'] = kg.apply(lambda row: name_to_index_dict[(row['y_type'], row['y_name'])], axis=1)
    
    return kg
    
def processNodeData(data, kg, embedding_dim):
        
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

def processEdgeData(data, kg):
    
    def getEdgeDict(kg): # Create edge dictionary to keep track of within-type indices
        
        # Drop duplicates
        reducedEdges = edges.drop_duplicates(subset=['x_index'], keep='first')
        
        # Create index within group for each node type and store in dictionary
        reducedEdges['group_idx'] = reducedEdges.groupby('x_type').cumcount()
        global_to_local_dict = dict(zip(reducedEdges['x_index'], reducedEdges['group_idx']))
        
        return global_to_local_dict

    # Get edge dictionary
    edges = deepcopy(kg)
    global_to_local_dict = getEdgeDict(edges)

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
        x_indices = torch.tensor(group['group_x'].values, dtype=torch.long)
        y_indices = torch.tensor(group['group_y'].values, dtype=torch.long)
        edge_list = torch.stack([x_indices, y_indices], dim=0)
            
        # Store in data
        data[group['x_type'].values[0], relation, group['y_type'].values[0]].edge_index = edge_list
    
    # Store edge dict for future use
    data['global_to_local_dict'] = global_to_local_dict
    
def processRelationData(indices):
    
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

def constructDiseaseSimilarity(data,k,device):
    
    # Generate and combine the one-hot vectors for all important conditions
    def generateOverallOneHot(disease_idx,data):
        
        # Generate a one-hot vector for a single condition
        def generateOneHot(node_idx, edge_type, data):
            
            # Get neighbors of the node
            edges = np.array(data[edge_type].edge_index)
            mask = np.where(edges[0] == node_idx)[0]
            neighbors = edges[1][mask]
            
            # Generate the one-hot vector
            one_hot = np.zeros(data[edge_type[2]].num_nodes,dtype=int)
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
    
def processData(embedding_dim, batch_size, fileName, device, k=10):
    
    # Get data path and process knowledge graph
    path = os.path.dirname(os.getcwd())
    kg = processFiles(path, fileName)
    
    # Create HeteroData object
    data = HeteroData()
    
    # Process node and edge data
    print("Processing node data...")
    processNodeData(data, kg, embedding_dim)
    print("Processing edge data...")
    processEdgeData(data, kg)    
    
    # Get dictionary of names to numbers for relations
    name_to_num = dict(zip(kg['relation'], pd.factorize(kg['relation'])[0]))
    
    # Get pretrain indices and process relation
    raw_pretrain_indices = kg[['x_index','x_type','relation','y_index']]
    raw_pretrain_indices['relation'] = raw_pretrain_indices['relation'].map(name_to_num)
    print("Setting up pretraining data...")
    pretrain_indices = processRelationData(raw_pretrain_indices)
    
    # Get finetuneindices and process relation
    raw_finetune_indices = kg[['x_index','x_type','relation','y_index']]
    raw_finetune_indices = raw_finetune_indices.loc[(raw_finetune_indices['relation'] == 'contraindication') | 
                                                    (raw_finetune_indices['relation'] == 'indication') | 
                                                    (raw_finetune_indices['relation'] == 'off_label_use')]
    raw_finetune_indices['relation'] = raw_finetune_indices['relation'].map(name_to_num)
    print("Setting up finetuning data...")
    finetune_indices = processRelationData(raw_finetune_indices)

    # Make sure indices are a compatible datatype
    pretrain_indices = pretrain_indices.astype({'x_index': 'int64', 'relation': 'int64', 'y_index': 'int64'})
    finetune_indices = finetune_indices.astype({'x_index': 'int64', 'relation': 'int64', 'y_index': 'int64'})
    
    # Turn into tensors
    pdata = torch.tensor(pretrain_indices.values,dtype=torch.long)
    fdata = torch.tensor(finetune_indices.values,dtype=torch.long)

    # Split data into train, validation, and test sets
    psplit = torch.utils.data.random_split(pdata, [0.8,0.2])
    fsplit = torch.utils.data.random_split(fdata, [0.8,0.1,0.1])

    # Create dataloaders
    ptrain_loader = DataLoader(psplit[0], batch_size=batch_size, shuffle=True)
    pval_loader = DataLoader(psplit[1], batch_size=batch_size, shuffle=False)

    ftrain_loader = DataLoader(fsplit[0], batch_size=batch_size, shuffle=True)
    fval_loader = DataLoader(fsplit[1], batch_size=batch_size, shuffle=False)
    ftest_loader = DataLoader(fsplit[2], batch_size=batch_size, shuffle=False)
    
    # Data must be undirected for proper GNN Message Passing
    data = T.ToUndirected()(data)
    
    # Save disease similarity matrix
    if 'disease_similarity.pt' not in os.listdir():
        print("Processing disease similarities...")
        torch.save(constructDiseaseSimilarity(data,k,device),'disease_similarity.pt')
        
    ## FUTURE MODIFICATION FOR OUR DISEASE INPUT
    data['similarity'] = torch.load('disease_similarity.pt')
    print("Data processing complete.")
    
    return data, ptrain_loader, pval_loader, ftrain_loader, fval_loader, ftest_loader, kg
