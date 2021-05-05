import networkx as nx
from typing import Union
from networkx.readwrite.json_graph import adjacency
from torch import Tensor
from numpy import ndarray
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform


def nx_second_order_proximity(
    G: Union[nx.Graph, nx.DiGraph],
    node_ids: Union[Tensor, ndarray, list],
    whole_graph_proximity: bool = True,
    distance_metric: str = "cosine",
    norm_distance: bool = True,
) -> ndarray:
    """
    Takes a networkx graph G and generates second-order node proximities, also known
    as structural equivalence relations.
    Nodes are similar, if they share similar ties to alters.

    Parameters
    ----------
    G : Union[nx.Graph,nx.DiGraph]
        Input graph
    node_ids : Union[Tensor,ndarray,list]
        List of nodes. Most exist in G.
    whole_graph_proximity : bool, optional
        If True, similarities between nodes in node_ids is computed based
        on all alters in the graph (including those not in node_ids).
        If False, similarities are only calculated based on nodes contained in
        node_ids, by default True
    distance_metric : str, optional
        Any distance metric from scipy.spatial.distance that works
        without parameter, by default 'cosine'
    norm_distance : bool, optional
        If true, distances are scaled such that the highest distance is 1.
        This implies that distances depend on the sample provided, by default True

    Returns
    -------
    ndarray
        Similarity matrix of dimension len(node_ids)^2
    """

    if isinstance(node_ids, list):
        node_ids = np.array(node_ids)

    if whole_graph_proximity:
        adjacency_matrix = np.zeros([len(G.nodes), len(G.nodes)])
        similarity_matrix = np.zeros([len(node_ids), len(G.nodes)])
    else:
        adjacency_matrix = np.zeros([len(node_ids), len(node_ids)])
        similarity_matrix = np.zeros([len(node_ids), len(node_ids)])

    

    if whole_graph_proximity:
        for i, node in enumerate(node_ids):
            for j, (alter, datadict) in enumerate(G.adj[node].items()):
                if hasattr(datadict, "weight"):
                    weight = datadict["weight"]
                else:
                    weight = 1
                adjacency_matrix[i, alter] = weight
    else:
        G_sub = G.subgraph(node_ids)
        for i, node in enumerate(node_ids):
            for j, (alter, datadict) in enumerate(G_sub[node].items()):
                if hasattr(datadict, "weight"):
                    weight = datadict["weight"]
                else:
                    weight = 1
                adjacency_matrix[i, j] = weight

    similarity_matrix = pdist(adjacency_matrix, metric=distance_metric)
    if norm_distance:
        similarity_matrix = similarity_matrix / np.max(
            similarity_matrix
        )  # Norm max distance to 1
    similarity_matrix = 1 - squareform(similarity_matrix)
    similarity_matrix = np.nan_to_num(similarity_matrix, copy=False)

    return similarity_matrix
