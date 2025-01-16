import torch
from torch import Tensor
from torch.nn import Module, SiLU, Parameter
import torch.nn.functional as F
from torch_geometric import nn
from torch_geometric.nn import GATConv, to_hetero
from torch_geometric.data import HeteroData
from torch_geometric.nn import Sequential as GeoSequential
from typing import Tuple
import math

class KGLinkPredictor(Module):
    """
        Knowledge Graph Link Predictor
    """
    
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int, 
                 data: HeteroData,
                 num_heads: int=1, 
                 num_layers: int=2, 
                ):
        
        super(KGLinkPredictor, self).__init__()
        
        ## Build encoder
        layers = []

        # Input layer
        layers.append((GATConv(in_channels, hidden_channels, add_self_loops=False, heads=num_heads), 'x, edge_index -> x'))
        
        # Intermediate layers
        for _ in range(num_layers - 1):
            layers.append((SiLU(inplace=True), 'x -> x'))
            layers.append((GATConv(hidden_channels * num_heads, hidden_channels, add_self_loops=False, heads=num_heads), 'x, edge_index -> x'))

        # Package layers and convert to hetero
        self.Encoder = GeoSequential('x, edge_index', layers)
        self.Encoder = to_hetero(self.Encoder, data.metadata())
        
        ## Build decoder
        self.Decoder = DistMultMod(data, hidden_channels, num_heads)
        
        # Save data
        self.data = data

    def forward(self, head_indices: Tensor, relations: Tensor, tail_indices: Tensor) -> Tensor:
        
        # Update embeddings
        self.update()
        
        # Return prediction
        return torch.sigmoid(self.Decoder(head_indices, relations, tail_indices))
    
    def loss(self, head_index: Tensor, relation: Tensor, tail_index: Tensor) -> Tensor:
        return self.Decoder.loss(head_index, relation, tail_index)
    
    def update(self) -> None:
        
        # Message Passing
        x = self.Encoder(self.data.x_dict,self.data.edge_index_dict)
        
        # Update Embeddings
        self.Decoder.node_emb = torch.vstack([*x.values()])

class DistMultMod(torch.nn.Module):
    """
        Modified DistMult Model
    """

    def __init__(
        self,
        data: HeteroData,
        hidden_channels: int,
        num_heads: int = 1,
        margin: float = 1.0,
        sparse: bool = False,
    ):
        super().__init__()
        
        self.num_nodes = data.num_nodes
        self.num_relations = data.num_edges
        self.hidden_channels = hidden_channels

        self.node_emb = torch.empty(self.num_nodes, hidden_channels*num_heads)
        self.rel_emb = Parameter(torch.empty(self.num_relations, hidden_channels*num_heads))
        
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
        loss_tag: bool=True
    ) -> Tensor:
        
        for i,relation in enumerate(rel_types):
            
            # Check if relation is a drug-disease relation
            if loss_tag and (relation == 2 or relation == 3 or relation == 4):
                
                # Get head index and modify disease embedding
                head_index = head_indices[i].item()
                self.node_emb[head_index] = self.modifyDiseaseEmbedding(head_index, relation)
        
        head = self.node_emb[head_indices]
        tail = self.node_emb[tail_indices]
        rel = self.rel_emb[rel_types]
        
        return (head * rel * tail).sum(dim=-1)
    
    def modifyDiseaseEmbedding(self, head_index: int, relation: int, l: float=0.7) -> Tensor:
        
        # Get proper relation title for accessing dataset
        relation_names = ['contraindication', 'indication', 'off_label_use']
        relation_name = relation_names[relation-2]
        relation_title = ('disease',relation_name,'drug')
        
        # Adjust head index to get the correct disease index
        adjusted_head_index = self.data['global_to_local_dict'][head_index]

        # Get the disease constant
        degree_in_relation = (self.data[relation_title].edge_index[0] == adjusted_head_index).sum().item()
        disease_constant = l * math.exp(-l * degree_in_relation) + 0.2      
    
        # Get most similar neighbors and initialize disease modification vector
        similarity_scores, neighbors = self.data['similarity'][adjusted_head_index]
        neighbor_embeddings = self.node_emb[neighbors.long()]

        # Compute weighted sum of neighbor embeddings
        disease_vector = torch.sum(similarity_scores.unsqueeze(1) * neighbor_embeddings, dim=0, keepdim=True)

        # Return modified embedding
        return disease_constant * disease_vector + (1 - disease_constant) * self.node_emb[head_index]

    # Margin Ranking Loss
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