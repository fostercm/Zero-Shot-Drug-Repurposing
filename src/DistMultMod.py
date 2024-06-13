import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding
from torch_geometric.nn.kge import KGEModel
from torch_geometric.data import HeteroData
from typing import Tuple


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
        margin: float = 1.0,
        sparse: bool = False,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_channels = hidden_channels

        self.node_emb = torch.empty(num_nodes, hidden_channels)
        self.rel_emb = torch.empty(num_relations, hidden_channels)
        self.data = data

        self.margin = margin

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb)


    def forward(
        self,
        head_index: Tensor,
        rel_type: Tensor,
        tail_index: Tensor,
    ) -> Tensor:

        head = torch.empty(len(head_index), self.hidden_channels)
        tail = torch.empty(len(tail_index), self.hidden_channels)
        
        keys = list(self.data.edge_index_dict.keys())
        for i in range(len(rel_type)):
            head[i][:] = self.node_emb[keys[rel_type[i]][0]][head_index[i]]
            tail[i][:] = self.node_emb[keys[rel_type[i]][2]][tail_index[i]]
        rel = self.rel_emb[rel_type]
        return (head * rel * tail).sum(dim=-1)

    def loss(
            self,
            head_index: Tensor,
            rel_type: Tensor,
            tail_index: Tensor,
        ) -> Tensor:

        pos_score = self(head_index, rel_type, tail_index)
        neg_score = self(*self.random_sample(head_index, rel_type, tail_index))

        return F.margin_ranking_loss(
            pos_score,
            neg_score,
            target=torch.ones_like(pos_score),
            margin=self.margin,
        )
    
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
