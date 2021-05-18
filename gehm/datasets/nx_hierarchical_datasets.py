from torch.utils.data import Dataset
import networkx as nx
from typing import Union
import torch
import numpy as np
from gehm.model.distances import matrix_cosine
from gehm.utils.funcs import row_norm
import logging

class nx_hierarchical_dataset(Dataset):
    """Dataset from networkX Graph without hierarchies for standard SDNE"""

    def __init__(
            self,
            G: Union[nx.Graph, nx.DiGraph],
            hierarchy_dict:dict = None, norm_rows: bool = True,
    ):
        self.nodes = np.array(list(G.nodes))


        # Create Node labels
        self.node_idx = np.linspace(0, len(self.nodes), len(self.nodes), False, dtype=int)
        self.node_idx_dict = dict(zip(self.node_idx, self.nodes))
        self.rev_node_idx_dict = {v: k for k, v in self.node_idx_dict.items()}
        self.node_idx = torch.LongTensor(self.node_idx)
        self.node_idx.requires_grad = False

        if hierarchy_dict is None:
            hierarchy_dict={}
            try:
                for node in self.nodes:
                    hierarchy_dict[self.rev_node_idx_dict[node]]=G.nodes[node]['hierarchy']
            except:
                message="No hierarchy dictionary provides and none found in Graph!"
                logging.error(message)
                raise RuntimeError(message)
        else:
            hierarchy_dict_idx={}
            try:
                for node in hierarchy_dict.keys():
                    hierarchy_dict_idx[self.rev_node_idx_dict[node]]=hierarchy_dict[node]
            except:
                message="Could not parse hierarchy dict"
                logging.error(message)
                raise RuntimeError(message)           
            hierarchy_dict = hierarchy_dict_idx

        self.hierarchy_dict=hierarchy_dict
        self.hierarchy_vals=np.unique(list(hierarchy_dict.values()))
        self.nr_hierarchies = len(self.hierarchy_vals)



        # Derive node similarities in whole graph

        adjacency_matrix = torch.as_tensor(np.array(nx.adjacency_matrix(G, weight="weight").todense()),dtype=float)
        self.sim1 = row_norm(adjacency_matrix)
        self.sim1.requires_grad = False

        self.sim2 = matrix_cosine(self.sim1)
        self.sim2 = row_norm(self.sim2)
        self.sim2.requires_grad = False

        # Derive hierarchy attention matrix

        self.hierarchy = torch.zeros(self.sim1.shape)
        self.hierarchy.requires_grad = False

        for node in self.node_idx:
            for peer in self.node_idx:
                if self.hierarchy_dict[int(node)]>=self.hierarchy_dict[int(peer)]:
                    self.hierarchy[node,peer]=1

        


    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        node = self.node_idx[idx]
        hierarchy = torch.as_tensor(self.hierarchy_dict[idx])
        h_attention = self.hierarchy[idx, :].float()
        similarity1 = self.sim1[idx, :].float()



        similarity2 = self.sim2[idx, :].float()

        return node, hierarchy, h_attention,similarity1, similarity2
