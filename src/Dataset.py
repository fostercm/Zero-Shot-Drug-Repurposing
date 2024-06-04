import pandas as pd
import os
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch

def processData(batch_size):
    
    # Get data path and import important files
    path = os.path.dirname(os.getcwd())
    
    nodes = pd.read_csv(path + r'\data\nodes.csv')
    kg = pd.read_csv(path + r'\data\kg.csv')
    
    # Create HeteroData object
    data = HeteroData()
    
    # Get the amount of nodes for each node type
    node_dict = {}

    for node in nodes['node_type']:
        node_dict[node] = node_dict.get(node, 0) + 1
        
    # Add nodes to HeteroData object with randomized features
    for node_type in node_dict.keys():
        data[node_type].x = torch.randn(node_dict[node_type], 16)
    data['num_nodes'] = nodes.shape[0]
    
    # Add edges to HeteroData object
    for edge_type in kg['relation'].unique():
        relation_df = kg.loc[kg['relation'] == edge_type]
        t1 = relation_df['x_type'].unique()[0]
        t2 = relation_df['y_type'].unique()[0]
        edge_list = [[],[]]
        for edge_subtype in relation_df['display_relation'].unique():
            sub_relation_df = relation_df.loc[relation_df['display_relation'] == edge_subtype]
            if sub_relation_df['x_type'].unique()[0] == t1:
                edge_list[0].extend(sub_relation_df['x_index'].to_list())
                edge_list[1].extend(sub_relation_df['y_index'].to_list())
            else:
                edge_list[0].extend(sub_relation_df['y_index'].to_list())
                edge_list[1].extend(sub_relation_df['x_index'].to_list())
        edge_list = torch.Tensor(edge_list)
        data[t1, edge_type, t2].edge_index = edge_list
    data['num_edges'] = kg.shape[0] 
    
    # Make pretraining and finetuning datasets
    pretrain_head_idx = []
    pretrain_relation = []
    pretrain_tail_idx = []

    finetine_head_idx = []
    finetine_relation = []
    finetine_tail_idx = []

    for i,edge_type in enumerate(data.edge_types):
        pretrain_head_idx.extend(data[edge_type].edge_index[0].tolist())
        pretrain_relation.extend([i]*data[edge_type].edge_index.shape[1])
        pretrain_tail_idx.extend(data[edge_type].edge_index[1].tolist())
        
        if edge_type[1] == 'contraindication':
            finetine_head_idx.extend(data[edge_type].edge_index[0].tolist())
            finetine_relation.extend([0]*data[edge_type].edge_index.shape[1])
            finetine_tail_idx.extend(data[edge_type].edge_index[1].tolist())
        elif edge_type[1] == 'indication':
            finetine_head_idx.extend(data[edge_type].edge_index[0].tolist())
            finetine_relation.extend([1]*data[edge_type].edge_index.shape[1])
            finetine_tail_idx.extend(data[edge_type].edge_index[1].tolist())

    pretrain_head_idx = torch.tensor(pretrain_head_idx)
    pretrain_relation = torch.tensor(pretrain_relation)
    pretrain_tail_idx = torch.tensor(pretrain_tail_idx)

    finetine_head_idx = torch.tensor(finetine_head_idx)
    finetine_relation = torch.tensor(finetine_relation)
    finetine_tail_idx = torch.tensor(finetine_tail_idx)
    
    pdata = torch.stack([pretrain_head_idx, pretrain_relation, pretrain_tail_idx], dim=1)
    fdata = torch.stack([finetine_head_idx, finetine_relation, finetine_tail_idx], dim=1)

    psplit = torch.utils.data.random_split(pdata, [0.8,0.1,0.1])
    fsplit = torch.utils.data.random_split(fdata, [0.8,0.1,0.1])

    ptrain_loader = DataLoader(psplit[0], batch_size=batch_size, shuffle=True)
    pval_loader = DataLoader(psplit[1], batch_size=batch_size, shuffle=True)
    ptest_loader = DataLoader(psplit[2], batch_size=batch_size, shuffle=True)

    ftrain_loader = DataLoader(fsplit[0], batch_size=batch_size, shuffle=True)
    fval_loader = DataLoader(fsplit[1], batch_size=batch_size, shuffle=True)
    ftest_loader = DataLoader(fsplit[2], batch_size=batch_size, shuffle=True)
    
    return data, ptrain_loader, pval_loader, ptest_loader, ftrain_loader, fval_loader, ftest_loader