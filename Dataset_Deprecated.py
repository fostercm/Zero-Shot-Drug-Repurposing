import pandas as pd
import os
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch
from copy import deepcopy
from torch.nn.init import xavier_uniform_

def processData(embedding_dim, batch_size):
    
    # Get data path and import important files
    path = os.path.dirname(os.getcwd())
    
    kg = pd.read_csv(path + r'/data/kg.csv')
    kg['x_type']= kg['x_type'].apply(lambda x: x.replace("/","_"))
    kg['y_type']= kg['y_type'].apply(lambda x: x.replace("/","_"))
    kg['relation']= kg['relation'].apply(lambda x: x.replace("-","_"))
    kg['relation']= kg['relation'].apply(lambda x: x.replace(" ","_"))
    
    # Create HeteroData object
    data = HeteroData()
    
    # Get the amount of nodes for each node type
    nodes = deepcopy(kg)
    nodes.drop_duplicates(subset=['x_index'], keep='first', inplace=True)
    node_dict = {}

    for node_type in nodes['x_type'].unique():
        split = nodes.loc[kg['x_type'] == node_type]
        node_dict[node_type] = len(split['x_index'].unique())
        
    # Add nodes to HeteroData object with randomized features
    for node_type in node_dict.keys():
        data[node_type].x = torch.empty(node_dict[node_type], embedding_dim, requires_grad=True)
        xavier_uniform_(data[node_type].x)
    
    # Create edge dict to keep track of within-type indices
    temp = deepcopy(kg)
    temp.drop_duplicates(subset=['x_index'], keep='first', inplace=True)
    temp['group_idx'] = temp.groupby('x_type').cumcount()
    idx_to_group = dict(zip(temp['x_index'], temp['group_idx']))
    
    # Get edge connections and add to data object
    edges = deepcopy(kg)

    # Apply edge dictionary
    edges['group_x'] = edges['x_index'].map(idx_to_group)
    edges['group_y'] = edges['y_index'].map(idx_to_group)

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
    
    # Get indices and process relation
    raw_pretrain_indices = kg[['x_index','x_type','relation','y_index']]
    name_to_num = dict(zip(raw_pretrain_indices['relation'], pd.factorize(raw_pretrain_indices['relation'])[0]))
    raw_pretrain_indices['relation'] = raw_pretrain_indices['relation'].map(name_to_num)

    # Group indices by relation
    groups = raw_pretrain_indices.groupby('relation')
    pretrain_indices = pd.DataFrame(columns=['x_index','relation','y_index'])

    for relation, group in groups:
        
        # Get the main group
        subgroups = group.groupby('x_type')
        group_name = list(subgroups.groups)[0]
        group = subgroups.get_group(group_name)
        
        # Add only the main group
        pretrain_indices = pd.concat([pretrain_indices,group[['x_index','relation','y_index']]])

    # Get indices and process relation
    raw_finetune_indices = kg[['x_index','x_type','relation','y_index']]
    raw_finetune_indices = raw_finetune_indices.loc[(raw_finetune_indices['relation'] == 'contraindication') | 
                                                    (raw_finetune_indices['relation'] == 'indication') | 
                                                    (raw_finetune_indices['relation'] == 'off_label_use')]
    raw_finetune_indices['relation'] = raw_finetune_indices['relation'].map(name_to_num)

    # Group indices by relation
    groups = raw_finetune_indices.groupby('relation')
    finetune_indices = pd.DataFrame(columns=['x_index','relation','y_index'])

    for relation, group in groups:
        
        # Get the main group
        subgroups = group.groupby('x_type')
        group_name = list(subgroups.groups)[0]
        group = subgroups.get_group(group_name)
        
        # Add only the main group
        finetune_indices = pd.concat([finetune_indices,group[['x_index','relation','y_index']]])

    # Make sure indices are a compatible datatype
    pretrain_indices = pretrain_indices.astype({'x_index': 'int32', 'relation': 'int32', 'y_index': 'int32'})
    finetune_indices = finetune_indices.astype({'x_index': 'int32', 'relation': 'int32', 'y_index': 'int32'})
    
    # Turn into tensors
    pdata = torch.tensor(pretrain_indices.values,dtype=torch.long)
    fdata = torch.tensor(finetune_indices.values,dtype=torch.long)

    # Split data into train, validation, and test sets
    psplit = torch.utils.data.random_split(pdata, [0.8,0.1,0.1])
    fsplit = torch.utils.data.random_split(fdata, [0.8,0.1,0.1])

    # Create dataloaders
    ptrain_loader = DataLoader(psplit[0], batch_size=batch_size, shuffle=True)
    pval_loader = DataLoader(psplit[1], batch_size=batch_size, shuffle=True)
    ptest_loader = DataLoader(psplit[2], batch_size=batch_size, shuffle=True)

    ftrain_loader = DataLoader(fsplit[0], batch_size=batch_size, shuffle=True)
    fval_loader = DataLoader(fsplit[1], batch_size=batch_size, shuffle=True)
    ftest_loader = DataLoader(fsplit[2], batch_size=batch_size, shuffle=True)
    
    # Data must be undirected for proper GNN Message Passing
    data = T.ToUndirected()(data)
    
    return data, ptrain_loader, pval_loader, ptest_loader, ftrain_loader, fval_loader, ftest_loader, idx_to_group