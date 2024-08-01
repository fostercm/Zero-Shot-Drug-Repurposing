import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from typing import Tuple
import numpy as np
import math


class DistMultMod(torch.nn.Module):
    r"""The DistMult model from the `"Embedding Entities and Relations for
    Learning and Inference in Knowledge Bases"
    <https://arxiv.org/abs/1412.6575>`_ paper.

    :class:`DistMult` models relations as diagonal matrices, which simplifies
    the bi-linear interaction between the head and tail entities to the score
    function:

    .. math::
        d(h, r, t) = < \mathbf{e}_h,  \mathbf{e}_r, \mathbf{e}_t >

    .. note::

        For an example of using the :class:`DistMult` model, see
        `examples/kge_fb15k_237.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        kge_fb15k_237.py>`_.

    Args:
        num_nodes (int): The number of nodes/entities in the graph.
        num_relations (int): The number of relations in the graph.
        hidden_channels (int): The hidden embedding size.
        margin (float, optional): The margin of the ranking loss.
            (default: :obj:`1.0`)
        sparse (bool, optional): If set to :obj:`True`, gradients w.r.t. to
            the embedding matrices will be sparse. (default: :obj:`False`)
    """
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_channels: int,
        data: HeteroData,
        similarity_path: str,
        local_idx_map: dict,
        margin: float = 1.0,
        sparse: bool = False,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_channels = hidden_channels

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.node_emb = torch.empty(num_nodes, hidden_channels).to(self.device)
        self.rel_emb = torch.empty(num_relations, hidden_channels).to(self.device)
        self.data = data

        self.margin = margin
        self.similarity = torch.load(similarity_path)
        self.local_idx_map = local_idx_map

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb)


    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
        loss_tag: bool = True,
    ) -> Tensor:
        
        for i,relation in enumerate(rel_type):
            
            # Get the disease constant
            if loss_tag and (relation == 2 or relation == 3 or relation == 4):
                # Disease similarity vector
                disease_vector = torch.zeros(1, self.hidden_channels, requires_grad=True).to(self.device)
                
                # Adjust head index to get the correct disease index
                adjusted_head_index = self.local_idx_map[head_index[i].cpu().item()]
                relations = ['contraindication', 'indication', 'off_label_use']
                relation = ('disease',relations[relation.cpu().item()-2],'drug')
                disease_constant = self.get_disease_constant(adjusted_head_index, relation)         
                # top-k similarity
                neighbors = self.similarity[adjusted_head_index][1]
                for j,neighbor in enumerate(neighbors):
                    disease_vector += self.similarity[adjusted_head_index][0][j] * self.node_emb[int(neighbor.item())]
                # print(f"disease vector | {disease_vector}")
                
                self.node_emb[head_index[i]] = disease_constant * disease_vector + (1-disease_constant) * self.node_emb[head_index[i]]
                # self.node_emb[head_index[i]] = (1-disease_constant) * self.node_emb[head_index[i]]
        
        head = self.node_emb[head_index]
        tail = self.node_emb[tail_index]
        rel = self.rel_emb[rel_type]
        
        return (head * rel * tail).sum(dim=-1)
    
    def get_disease_constant(self, disease_idx, relation, l=0.7):
        relation_edge_list = self.data[relation].edge_index[0]
        relation_edge_list = relation_edge_list.cpu().numpy()
        degree_in_relation = len(np.where(relation_edge_list == disease_idx)[0])
        return l * math.exp(-l * degree_in_relation) + 0.2

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
