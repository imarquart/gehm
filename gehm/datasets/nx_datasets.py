from torch.utils.data import Dataset
import networkx as nx
from typing import Union
import torch
import numpy as np
from gehm.model.distances import matrix_cosine
from gehm.utils.funcs import row_norm
from gehm.utils.np_distances import (
    nx_second_order_proximity,
    nx_first_order_proximity,
    second_order_proximity,
)


class nx_dataset_onelevelOLD(Dataset):
    """Dataset from networkX Graph without hierarchies"""

    def __init__(
        self,
        G: Union[nx.Graph, nx.DiGraph],
        distance_metric: str = "cosine",
        norm_rows: bool = True,
        proximities: list = [1, 2],
    ):

        if not isinstance(proximities, list):
            proximities = [proximities]
        self.proximities = proximities
        self.nodes = np.array(list(G.nodes))

        # Create Node labels
        self.node_idx = np.linspace(0,len(self.nodes), len(self.nodes), False, dtype=int)
        self.node_idx_dict =dict(zip(self.nodes, self.node_idx))
        self.node_idx=torch.as_tensor(self.node_idx)

        # Derive node similarities in whole graph
        if 1 in proximities:
            self.sim1 = nx_first_order_proximity(
                G=G,
                node_ids=self.nodes,
                whole_graph_proximity=True,
                to_batch=False,
                distance_metric=distance_metric,
                norm_rows_in_sample=False,
                norm_rows=norm_rows,
            )
        else:
            self.sim1 = None

        if 2 in proximities:
            if self.sim1 is None:
                self.sim2 = nx_second_order_proximity(
                    G=G,
                    node_ids=self.nodes,
                    whole_graph_proximity=True,
                    to_batch=False,
                    distance_metric=distance_metric,
                    norm_rows_in_sample=False,
                    norm_rows=norm_rows,
                )
            else:
                self.sim2 = second_order_proximity(
                    self.sim1,
                    whole_graph_proximity=True,
                    to_batch=False,
                    distance_metric=distance_metric,
                    norm_rows_in_sample=False,
                    norm_rows=norm_rows,
                )
        else:
            self.sim2 = None

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):

        node = self.node_idx[idx].int()
        if 1 in self.proximities:
            similarity1 = self.sim1[idx, :].float()
        else:
            similarity1 = 0
        if 2 in self.proximities:
            similarity2 = self.sim2[idx, :].float()
        else:
            similarity2 = 0

        return node, similarity1, similarity2


class nx_dataset_onelevel(Dataset):
    """Dataset from networkX Graph without hierarchies"""

    def __init__(
        self,
        G: Union[nx.Graph, nx.DiGraph],
        distance_metric: str = "cosine",
        norm_rows: bool = True,
        proximities: list = [1, 2],
    ):

        if not isinstance(proximities, list):
            proximities = [proximities]
        self.proximities = proximities
        self.nodes = np.array(list(G.nodes))

        # Create Node labels
        self.node_idx = np.linspace(0,len(self.nodes), len(self.nodes), False, dtype=int)
        self.node_idx_dict =dict(zip(self.node_idx,self.nodes))
        self.node_idx=torch.LongTensor(self.node_idx)

        # Derive node similarities in whole graph

        self.sim1 = nx_first_order_proximity(
            G=G,
            node_ids=self.nodes,
            whole_graph_proximity=True,
            to_batch=False,
            distance_metric=distance_metric,
            norm_rows_in_sample=False,
            norm_rows=norm_rows,
        )


        self.sim2=matrix_cosine(self.sim1)
        self.sim2 = row_norm(self.sim2)
        self.sim2.requires_grad=False

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):

        node = self.node_idx[idx]
        similarity1 = self.sim1[idx, :].float()
        similarity2 = self.sim2[idx, :].float()


        return node, similarity1, similarity2

