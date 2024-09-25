import torch
from torch import Tensor
from torch.nn import Module, SiLU
import torch.nn.functional as F
from torch_geometric import nn
from torch_geometric.nn import GATConv, to_hetero
from torch_geometric.data import HeteroData
from typing import Tuple
from tqdm.notebook import tqdm
import math


class KGLinkPredictor(Module):
    def __init__(self, in_channels, hidden_channels, data):
        super(KGLinkPredictor, self).__init__()
        
        self.Encoder = nn.Sequential('x, edge_index', [
            (GATConv(in_channels,hidden_channels,add_self_loops=False), 'x, edge_index -> x'),
            SiLU(inplace=True),
            (GATConv(hidden_channels,hidden_channels,add_self_loops=False), 'x, edge_index -> x'),
        ])
        self.Encoder = to_hetero(self.Encoder, data.metadata())
        
        self.Decoder = DistMultMod(data, hidden_channels)
        
        self.data = data

    def forward(self, head_indices, relations, tail_indices):
        
        # Message Passing
        x = self.Encoder(self.data.x_dict,self.data.edge_index_dict)
        
        # Update Embeddings
        self.Decoder.node_emb = torch.vstack([*x.values()])
        
        # Return prediction
        return torch.sigmoid(self.Decoder(head_indices, relations, tail_indices))
    
    def loss(self, head_index, relation, tail_index):
        return self.Decoder.loss(head_index, relation, tail_index)

class DistMultMod(torch.nn.Module):

    def __init__(
        self,
        data: HeteroData,
        hidden_channels: int,
        margin: float = 1.0,
        sparse: bool = False,
    ):
        super().__init__()
        
        self.num_nodes = data.num_nodes
        self.num_relations = data.num_edges
        self.hidden_channels = hidden_channels

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.node_emb = torch.empty(self.num_nodes, hidden_channels).to(self.device)
        self.rel_emb = torch.empty(self.num_relations, hidden_channels).to(self.device)
        self.data = data

        self.margin = margin

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb)

    def forward(
        self,
        head_indices: Tensor,
        rel_types: Tensor,
        tail_indices: Tensor,
        loss_tag: bool = True,
    ) -> Tensor:
        
        
        for i,relation in enumerate(rel_types):
            
            # Check if relation is a drug-disease relation
            if loss_tag and (relation == 4 or relation == 5 or relation == 6):
                
                # Get head index and modify disease embedding
                head_index = head_indices[i].item()
                self.node_emb[head_index] = self.modifyDiseaseEmbedding(head_index, relation)
        
        head = self.node_emb[head_indices]
        tail = self.node_emb[tail_indices]
        rel = self.rel_emb[rel_types]
        
        return (head * rel * tail).sum(dim=-1)
    
    def modifyDiseaseEmbedding(self, head_index, relation, l=0.7):
        
        # Get proper relation title for accessing dataset
        relation_names = ['contraindication', 'indication', 'off_label_use']
        relation_name = relation_names[relation-4]
        relation_title = ('disease',relation_name,'drug')
        
        # Adjust head index to get the correct disease index
        adjusted_head_index = self.data['global_to_local_dict'][head_index]

        # Get the disease constant
        relation_edge_list = self.data[relation_title].edge_index[0]
        degree_in_relation = len(torch.where(relation_edge_list == adjusted_head_index)[0])
        disease_constant = l * math.exp(-l * degree_in_relation) + 0.2      
          
        # Get most similar neighbors and initialize disease modification vector
        disease_vector = torch.zeros(1, self.hidden_channels, requires_grad=True).to(self.device)
        neighbors = self.data['similarity'][adjusted_head_index][1]
        
        # Calculate disease modification vector using weighted neighbor embeddings
        for j,neighbor in enumerate(neighbors):
            disease_vector += self.data['similarity'][adjusted_head_index][0][j] * self.node_emb[int(neighbor.item())]
        
        # Return modified embedding
        return disease_constant * disease_vector + (1-disease_constant) * self.node_emb[head_index]

    def loss(
            self,
            head_index: Tensor,
            rel_type: Tensor,
            tail_index: Tensor,
        ) -> Tensor:


        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index), loss_tag=False)

        return F.margin_ranking_loss(pos_score,neg_score,target=torch.ones_like(pos_score),margin=self.margin)
    
    @torch.no_grad()
    def random_sample(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Randomly samples negative triplets by either replacing the head or
        the tail (but not both).

        Args:
            head_index (torch.Tensor): The head indices.
            rel_type (torch.Tensor): The relation type.
            tail_index (torch.Tensor): The tail indices.
        """
        # Random sample either `head_index` or `tail_index` (but not both):
        num_negatives = head_index.numel() // 2
        rnd_index = torch.randint(self.num_nodes, head_index.size(),
                                  device=head_index.device)

        head_index = head_index.clone()
        head_index[:num_negatives] = rnd_index[:num_negatives]
        tail_index = tail_index.clone()
        tail_index[num_negatives:] = rnd_index[num_negatives:]

        return head_index, rel_type, tail_index
    
def train(train_loader,val_loader, model, optimizer, device, epochs):
    
    # Set model to training mode
    model.train()
    
    for epoch in tqdm(range(epochs)):
        
        train_loss = 0
        for batch in train_loader:
            
            # Send data to GPU
            head_indices,relations,tail_indices = batch[:,0], batch[:,1], batch[:,2]
            head_indices,relations,tail_indices = head_indices.to(device),relations.to(device),tail_indices.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass through model to update node embeddings
            model(head_indices,relations,tail_indices)
            
            # Compute and backpropagate loss
            loss = model.loss(head_indices, relations, tail_indices)
            train_loss += loss.item()
            loss.backward()
            
            # Step optimizer
            optimizer.step()
        
        val_loss = 0
        for batch in val_loader:
            
            # Send data to GPU
            head_indices,relations,tail_indices = batch[:,0], batch[:,1], batch[:,2]
            head_indices,relations,tail_indices = head_indices.to(device),relations.to(device),tail_indices.to(device)
            
            # Forward pass through model to update node embeddings
            model(head_indices,relations,tail_indices)
            
            # Compute loss
            loss = model.loss(head_indices, relations, tail_indices)
            val_loss += loss.item()
                
        print(f"Epoch {epoch+1} | Average Training Loss: {train_loss / len(train_loader)} Average Validation Loss: {val_loss / len(val_loader)}")