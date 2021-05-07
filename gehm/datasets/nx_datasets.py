from torch.utils.data import Dataset, DataLoader
import networkx as nx
from typing import Union
import torch
import numpy as np

from gehm.utils.distances import (
    nx_second_order_proximity,
    nx_first_order_proximity,
    second_order_proximity,
)


class nx_dataset_onelevel(Dataset):
    """Dataset from networkX Graph without hierarchies"""

    def __init__(
        self,
        G: Union[nx.Graph, nx.DiGraph],
        distance_metric: str = "cosine",
        norm_rows: bool = True,
        proximities:list=[1,2]
    ):
        
        if not isinstance(proximities,list):
            proximities=[proximities]
        self.proximities=proximities
        self.nodes = np.array(list(G.nodes))

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
                self.sim2 = nx_second_order_proximity(G=G,node_ids=self.nodes,whole_graph_proximity=True, to_batch=False, distance_metric=distance_metric, norm_rows_in_sample=False, norm_rows=norm_rows)
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

        node = self.nodes[idx]
        if 1 in self.proximities:
            similarity1 = self.sim1[idx, :]
        else:
            similarity1 = 0
        if 2 in self.proximities:
            similarity2 = self.sim2[idx, :]
        else: 
            similarity2 = 0

        return node, similarity1,similarity2

