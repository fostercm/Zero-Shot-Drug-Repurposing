import pandas as pd
import os
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch
from copy import deepcopy

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
        data[node_type].x = torch.randn(node_dict[node_type], embedding_dim)
    
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
        
    # Data must be undirected for proper GNN Message Passing
    data = T.ToUndirected()(data)
    
    # Make pretraining and finetuning datasets
    pretrain_head_idx = []
    pretrain_relation = []
    pretrain_tail_idx = []

    finetune_head_idx = []
    finetine_relation = []
    finetine_tail_idx = []

    for i,edge_type in enumerate(data.edge_types):
        pretrain_head_idx.extend(data[edge_type].edge_index[0].tolist())
        pretrain_relation.extend([i]*data[edge_type].edge_index.shape[1])
        pretrain_tail_idx.extend(data[edge_type].edge_index[1].tolist())
        
        if edge_type[1] == 'contraindication' or edge_type[1] == 'rev_contraindication':
            finetune_head_idx.extend(data[edge_type].edge_index[0].tolist())
            finetine_relation.extend([0]*data[edge_type].edge_index.shape[1])
            finetine_tail_idx.extend(data[edge_type].edge_index[1].tolist())
        elif edge_type[1] == 'indication' or edge_type[1] == 'rev_indication':
            finetune_head_idx.extend(data[edge_type].edge_index[0].tolist())
            finetine_relation.extend([1]*data[edge_type].edge_index.shape[1])
            finetine_tail_idx.extend(data[edge_type].edge_index[1].tolist())
        elif edge_type[1] == 'off-label use' or edge_type[1] == 'rev_off-label use':
            finetune_head_idx.extend(data[edge_type].edge_index[0].tolist())
            finetine_relation.extend([2]*data[edge_type].edge_index.shape[1])
            finetine_tail_idx.extend(data[edge_type].edge_index[1].tolist())

    pretrain_head_idx = torch.tensor(pretrain_head_idx)
    pretrain_relation = torch.tensor(pretrain_relation)
    pretrain_tail_idx = torch.tensor(pretrain_tail_idx)

    finetune_head_idx = torch.tensor(finetune_head_idx)
    finetine_relation = torch.tensor(finetine_relation)
    finetine_tail_idx = torch.tensor(finetine_tail_idx)
    
    pdata = torch.stack([pretrain_head_idx, pretrain_relation, pretrain_tail_idx], dim=1)
    fdata = torch.stack([finetune_head_idx, finetine_relation, finetine_tail_idx], dim=1)

    psplit = torch.utils.data.random_split(pdata, [0.8,0.1,0.1])
    fsplit = torch.utils.data.random_split(fdata, [0.8,0.1,0.1])

    ptrain_loader = DataLoader(psplit[0], batch_size=batch_size, shuffle=True)
    pval_loader = DataLoader(psplit[1], batch_size=batch_size, shuffle=True)
    ptest_loader = DataLoader(psplit[2], batch_size=batch_size, shuffle=True)

    ftrain_loader = DataLoader(fsplit[0], batch_size=batch_size, shuffle=True)
    fval_loader = DataLoader(fsplit[1], batch_size=batch_size, shuffle=True)
    ftest_loader = DataLoader(fsplit[2], batch_size=batch_size, shuffle=True)
    
    return data, ptrain_loader, pval_loader, ptest_loader, ftrain_loader, fval_loader, ftest_loader