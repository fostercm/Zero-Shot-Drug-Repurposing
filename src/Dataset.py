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
    
    kg = pd.read_csv(path + r'\data\kg.csv')
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

    # There are inconsistencies in the data, so we need to make sure that the group indices are stored to the correct side of the edge list
    exceptions = ['pathway_protein', 'drug_effect', 'drug_protein', 'exposure_molfunc', 'exposure_protein', 'molfunc_protein']

    # Group by relation
    groups = edges.groupby('relation')

    for relation, group in groups:
        
        # Done to see if the group is consistent
        subgroups = group.groupby('x_type')
        
        if subgroups.ngroups == 1:
            
            # If consistent, make a simple edge list
            x_indices = torch.tensor(group['group_x'].values, dtype=torch.long)
            y_indices = torch.tensor(group['group_y'].values, dtype=torch.long)
        
            edge_list = torch.stack([x_indices, y_indices], dim=0)
            
        else:
            
            # If not, then we need to make sure that the group indices are stored to the correct side of the edge list
            edge_list = [[],[]]
            
            if relation in exceptions:
                edge_list[0].extend(list(subgroups.get_group(list(subgroups.groups)[0])['group_x']))
                edge_list[1].extend(list(subgroups.get_group(list(subgroups.groups)[0])['group_y']))
                
                edge_list[0].extend(list(subgroups.get_group(list(subgroups.groups)[1])['group_y']))
                edge_list[1].extend(list(subgroups.get_group(list(subgroups.groups)[1])['group_x']))
                
            else:
                edge_list[0].extend(list(subgroups.get_group(list(subgroups.groups)[0])['group_y']))
                edge_list[1].extend(list(subgroups.get_group(list(subgroups.groups)[0])['group_x']))
                
                edge_list[0].extend(list(subgroups.get_group(list(subgroups.groups)[1])['group_x']))
                edge_list[1].extend(list(subgroups.get_group(list(subgroups.groups)[1])['group_y']))

            # Convert to useable type
            edge_list = torch.Tensor(edge_list)
            edge_list = edge_list.type(torch.int64)
            
        # Store in data
        data[group['x_type'].values[0], relation, group['y_type'].values[0]].edge_index = edge_list
    
    # Make pretraining and finetuning datasets
    pretrain_indices = kg[['x_index','relation','y_index']]
    name_to_num = dict(zip(pretrain_indices['relation'], pd.factorize(pretrain_indices['relation'])[0]))
    pretrain_indices['relation'] = pretrain_indices['relation'].map(name_to_num)

    finetune_indices = kg[['x_index','relation','y_index']]
    finetune_indices = finetune_indices.loc[(finetune_indices['relation'] == 'contraindication') | 
                                            (finetune_indices['relation'] == 'indication') | 
                                            (finetune_indices['relation'] == 'off_label_use')]
    finetune_indices['relation'] = finetune_indices['relation'].map(name_to_num)
    
    pdata = torch.tensor(pretrain_indices.values,dtype=torch.long)
    fdata = torch.tensor(finetune_indices.values,dtype=torch.long)

    psplit = torch.utils.data.random_split(pdata, [0.8,0.1,0.1])
    fsplit = torch.utils.data.random_split(fdata, [0.8,0.1,0.1])

    ptrain_loader = DataLoader(psplit[0], batch_size=batch_size, shuffle=True)
    pval_loader = DataLoader(psplit[1], batch_size=batch_size, shuffle=True)
    ptest_loader = DataLoader(psplit[2], batch_size=batch_size, shuffle=True)

    ftrain_loader = DataLoader(fsplit[0], batch_size=batch_size, shuffle=True)
    fval_loader = DataLoader(fsplit[1], batch_size=batch_size, shuffle=True)
    ftest_loader = DataLoader(fsplit[2], batch_size=batch_size, shuffle=True)
    
    # Data must be undirected for proper GNN Message Passing
    data = T.ToUndirected()(data)
    
    return data, ptrain_loader, pval_loader, ptest_loader, ftrain_loader, fval_loader, ftest_loader