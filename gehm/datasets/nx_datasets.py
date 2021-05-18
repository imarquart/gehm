from torch.utils.data import Dataset
import networkx as nx
from typing import Union
import torch
import numpy as np
from gehm.model.distances import matrix_cosine
from gehm.utils.funcs import row_norm


class nx_dataset_sdne(Dataset):
    """Dataset from networkX Graph without hierarchies for standard SDNE"""

    def __init__(
            self,
            G: Union[nx.Graph, nx.DiGraph],
            norm_rows: bool = True,
    ):
        self.nodes = np.array(list(G.nodes))

        # Create Node labels
        self.node_idx = np.linspace(0, len(self.nodes), len(self.nodes), False, dtype=int)
        self.node_idx_dict = dict(zip(self.node_idx, self.nodes))
        self.node_idx = torch.LongTensor(self.node_idx)
        self.node_idx.requires_grad = False

        # Derive node similarities in whole graph

        adjacency_matrix = torch.as_tensor(np.array(nx.adjacency_matrix(G, weight="weight").todense()),dtype=float)
        self.sim1 = row_norm(adjacency_matrix)
        self.sim1.requires_grad = False

        self.sim2 = matrix_cosine(self.sim1)
        self.sim2 = row_norm(self.sim2)
        self.sim2.requires_grad = False

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        node = self.node_idx[idx]
        similarity1 = self.sim1[idx, :].float()
        similarity2 = self.sim2[idx, :].float()

        return node, similarity1, similarity2



class nx_dataset_tsne(nx_dataset_sdne):
    """Dataset from networkX Graph without hierarchies for transformer SDNE"""
    def __init__(self, G: Union[nx.Graph, nx.DiGraph], norm_rows: bool = True):
        super().__init__(G, norm_rows)
    def __getitem__(self, idx):
        node = self.node_idx[idx]
        similarity1 = self.sim1[idx, :].float()
        similarity2 = self.sim2[idx, :].float()

        return node, similarity1, similarity2



class batch_nx_dataset_tsne(nx_dataset_sdne):
    """Dataset from networkX Graph without hierarchies for transformer SDNE"""
    def __init__(self, G: Union[nx.Graph, nx.DiGraph], norm_rows: bool = True, neighborhood_size:int=5):
        
        assert neighborhood_size <= len(G)-1, "Neighborhood size must be smaller than the number of nodes in the network!"
        
        self.neighborhood_size=neighborhood_size

        super().__init__(G, norm_rows)
    def __getitem__(self, idx):
        similarity1 = self.sim1[idx, :].float()
        node = self.node_idx[idx]
        proximate_peers = torch.argsort(similarity1, dim=-1, descending=True)[0:self.neighborhood_size]
        nodes=torch.cat([node.unsqueeze(0),proximate_peers])
        
        node = self.node_idx[nodes]
        similarity1 = self.sim1[nodes, :].float()
        similarity2 = self.sim2[nodes, :].float()

        return node, similarity1, similarity2