import networkx as nx
from typing import Union
from networkx.readwrite.json_graph import adjacency
from torch import Tensor
from numpy import ndarray
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import torch
from torch.nn.functional import cosine_similarity
from torch import cdist
from gehm.utils.funcs import row_norm


def second_order_proximity(
    adjacency_matrix: Union[Tensor, ndarray],
    indecies: Union[Tensor, ndarray, list] = None,
    whole_graph_proximity: bool = True,
    to_batch: bool = False,
    distance_metric: str = "cosine",
    norm_rows_in_sample: bool = False,
    norm_rows: bool = True,
) -> Tensor:
    """
    Takes an adjacency matrix and generates second-order node proximities, also known
    as structural equivalence relations.
    Nodes are similar, if they share similar ties to alters.

    Diagonal elements are set to zero.

    Note that this includes non-PyTorch operations!

    Parameters
    ----------
    adjacency_matrix: Union[Tensor, ndarray]
        Input adjacency_matrix
    indecies : Union[Tensor,ndarray,list]
        List of node indecies to consider in the matrix
    whole_graph_proximity : bool, optional
        If True, similarities between nodes in indecies is computed based
        on all alters in the matrix (including those not in indecies)
        If False, similarities are only calculated based on nodes contained in
        indecies.
    to_batch : bool, optional
        If true, will remove the row entries of nodes not in indecies
        If norm_rows is True, will also re-norm the rows, by default True
    distance_metric : str, optional
        Any distance metric from scipy.spatial.distance that works
        without parameter, by default 'cosine'
    norm_rows_in_sample : bool, optional
        If True, distances are scaled such that the highest distance is 1.
        This implies that distances depend on the sample provided, by default False
    norm_rows: bool, optional
        If True, distances are scaled for each node, such that sum(a_ij)=1
        This does not take into account the similarity to itself, a_ii, which is always 0.
    Returns
    -------
    ndarray
        Similarity matrix of dimension len(node_ids)^2
    """
    if indecies is None:
        indecies = np.arange(0, adjacency_matrix.shape[0])
    else:
        if isinstance(indecies, list):
            indecies = np.array(indecies)
        if isinstance(indecies, Tensor):
            indecies = indecies.numpy()
    if isinstance(adjacency_matrix, Tensor):
        adjacency_matrix = adjacency_matrix.numpy()

    if not whole_graph_proximity:
        adjacency_matrix = adjacency_matrix[indecies, :]
        adjacency_matrix = adjacency_matrix[:, indecies]

    similarity_matrix = pdist(adjacency_matrix, metric=distance_metric)
    similarity_matrix = 1 - squareform(similarity_matrix)
    similarity_matrix = similarity_matrix - np.eye(
        similarity_matrix.shape[0], similarity_matrix.shape[1]
    )
    if norm_rows_in_sample:
        similarity_matrix = similarity_matrix / np.max(
            similarity_matrix
        )  # Norm max similarity within the sample to 1
    if norm_rows and not to_batch:
        similarity_matrix = row_norm(similarity_matrix)
    similarity_matrix = np.nan_to_num(similarity_matrix, copy=False)
    if whole_graph_proximity:
        similarity_matrix = similarity_matrix[indecies, :]
        if to_batch:
            similarity_matrix = whole_graph_rows_to_batch(
                similarity_matrix, indecies, norm_rows=norm_rows
            )

    return torch.as_tensor(similarity_matrix)


def nx_first_order_proximity(
    G: Union[nx.Graph, nx.DiGraph],
    node_ids: Union[Tensor, ndarray, list],
    whole_graph_proximity: bool = True,
    to_batch: bool = False,
    distance_metric: str = "cosine",
    norm_rows_in_sample: bool = False,
    norm_rows: bool = True,
) -> Tensor:
    """
    Takes a networkx graph G and generates first-order node proximities.

    Diagonal elements are set to zero.

    Note that this includes non-PyTorch operations!

    Parameters
    ----------
    G : Union[nx.Graph,nx.DiGraph]
        Input graph
    node_ids : Union[Tensor,ndarray,list]
        List of nodes. Must exist in G.
    whole_graph_proximity : bool, optional
        If True, similarities between nodes in node_ids is computed based
        on all alters in the graph (including those not in node_ids)
        If False, similarities are only calculated based on nodes contained in
        node_ids.
        ATTN: Note that if True, ordering of rows reflects G.nodes
        if False, ordering reflects node_ids supplied (subnetwork)
        by default True
    to_batch : bool, optional
        If true, will remove the row entries of nodes not in node_list
        If norm_rows is True, will also re-norm the rows, by default True
    distance_metric : str, optional
        Any distance metric from scipy.spatial.distance that works
        without parameter, by default 'cosine'
    norm_rows_in_sample : bool, optional
        If True, distances are scaled such that the highest distance is 1.
        This implies that distances depend on the sample provided, by default False
    norm_rows: bool, optional
        If True, distances are scaled for each node, such that sum(a_ij)=1
        This does not take into account the similarity to itself, a_ii, which is always 0.
    Returns
    -------
    ndarray
        Similarity matrix of dimension len(node_ids)^2
    """

    if isinstance(node_ids, list):
        node_ids = np.array(node_ids)
    if isinstance(node_ids, Tensor):
        node_ids = node_ids.numpy()

    if whole_graph_proximity:
        adjacency_matrix = np.zeros([len(G.nodes), len(G.nodes)])
    else:
        adjacency_matrix = np.zeros([len(node_ids), len(node_ids)])

    if whole_graph_proximity:
        adjacency_matrix = np.array(nx.adjacency_matrix(G, weight="weight").todense())
    else:
        G_sub = G.subgraph(node_ids)
        for i, node in enumerate(node_ids):
            for j, (alter, datadict) in enumerate(G_sub[node].items()):
                if hasattr(datadict, "weight"):
                    weight = datadict["weight"]
                else:
                    weight = 1
                adjacency_matrix[i, j] = weight

    if norm_rows_in_sample:
        adjacency_matrix = adjacency_matrix / np.max(
            adjacency_matrix
        )  # Norm max similarity within the sample to 1
    if norm_rows and not to_batch:
        adjacency_matrix = row_norm(adjacency_matrix)
    adjacency_matrix = np.nan_to_num(adjacency_matrix, copy=False)
    if whole_graph_proximity:
        selection = np.searchsorted(np.array(G.nodes), node_ids)
        assert (
            np.array(G.nodes)[selection] == node_ids
        ).all(), "Internal error, subsetting nodes"
        adjacency_matrix = adjacency_matrix[selection, :]
        if to_batch:
            adjacency_matrix = whole_graph_rows_to_batch(
                adjacency_matrix, selection, norm_rows=norm_rows
            )

    return torch.as_tensor(adjacency_matrix)


def nx_second_order_proximity(
    G: Union[nx.Graph, nx.DiGraph],
    node_ids: Union[Tensor, ndarray, list],
    whole_graph_proximity: bool = True,
    to_batch: bool = False,
    distance_metric: str = "cosine",
    norm_rows_in_sample: bool = False,
    norm_rows: bool = True,
) -> Tensor:
    """
    Takes a networkx graph G and generates second-order node proximities, also known
    as structural equivalence relations.
    Nodes are similar, if they share similar ties to alters.

    Diagonal elements are set to zero.

    Note that this includes non-PyTorch operations!

    Parameters
    ----------
    G : Union[nx.Graph,nx.DiGraph]
        Input graph
    node_ids : Union[Tensor,ndarray,list]
        List of nodes. Must exist in G.
    whole_graph_proximity : bool, optional
        If True, similarities between nodes in node_ids is computed based
        on all alters in the graph (including those not in node_ids)
        If False, similarities are only calculated based on nodes contained in
        node_ids.
        ATTN: Note that if True, ordering of rows reflects G.nodes
        if False, ordering reflects node_ids supplied (subnetwork)
        by default True
    to_batch : bool, optional
        If true, will remove the row entries of nodes not in node_list
        If norm_rows is True, will also re-norm the rows, by default True
    distance_metric : str, optional
        Any distance metric from scipy.spatial.distance that works
        without parameter, by default 'cosine'
    norm_rows_in_sample : bool, optional
        If True, distances are scaled such that the highest distance is 1.
        This implies that distances depend on the sample provided, by default False
    norm_rows: bool, optional
        If True, distances are scaled for each node, such that sum(a_ij)=1
        This does not take into account the similarity to itself, a_ii, which is always 0.
    Returns
    -------
    ndarray
        Similarity matrix of dimension len(node_ids)^2
    """

    if isinstance(node_ids, list):
        node_ids = np.array(node_ids)
    if isinstance(node_ids, Tensor):
        node_ids = node_ids.numpy()

    if whole_graph_proximity:
        adjacency_matrix = np.zeros([len(G.nodes), len(G.nodes)])
        similarity_matrix = np.zeros([len(node_ids), len(G.nodes)])
    else:
        adjacency_matrix = np.zeros([len(node_ids), len(node_ids)])
        similarity_matrix = np.zeros([len(node_ids), len(node_ids)])

    if whole_graph_proximity:
        adjacency_matrix = nx.adjacency_matrix(G, weight="weight").todense()
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
    similarity_matrix = 1 - squareform(similarity_matrix)
    similarity_matrix = similarity_matrix - np.eye(
        similarity_matrix.shape[0], similarity_matrix.shape[1]
    )
    if norm_rows_in_sample:
        similarity_matrix = similarity_matrix / np.max(
            similarity_matrix
        )  # Norm max similarity within the sample to 1
    if norm_rows and not to_batch:
        similarity_matrix = row_norm(similarity_matrix)
    similarity_matrix = np.nan_to_num(similarity_matrix, copy=False)
    if whole_graph_proximity:
        selection = np.searchsorted(np.array(G.nodes), node_ids)
        assert (
            np.array(G.nodes)[selection] == node_ids
        ).all(), "Internal error, subsetting nodes"
        similarity_matrix = similarity_matrix[selection, :]
        if to_batch:
            similarity_matrix = whole_graph_rows_to_batch(
                similarity_matrix, selection, norm_rows=norm_rows
            )

    return torch.as_tensor(similarity_matrix)


def whole_graph_rows_to_batch(
    similarity_matrix: Union[Tensor, ndarray],
    indecies: Union[Tensor, ndarray, list],
    norm_rows: bool = True,
) -> Tensor:
    """
    Sorts matrix according to indecies and row-normalizes if desired

    Parameters
    ----------
    similarity_matrix : Union[Tensor,ndarray]
        input
    indecies : Union[Tensor, ndarray, list]
        indecies with order
    norm_rows : bool, optional
        whether to row norm, by default True

    Returns
    -------
    Tensor
        similarity_matrix
    """
    similarity_matrix = similarity_matrix[:, indecies]
    if norm_rows:
        similarity_matrix = row_norm(similarity_matrix)
    return torch.as_tensor(similarity_matrix)


def embedding_first_order_proximity(
    positions: Tensor, norm_rows: bool = True,
) -> Tensor:
    """
    A simple application of euclidian distance to the positions vector in the embedding space.
    Includes row normalization to get relative distances.

    Parameters
    ----------
    positions : Tensor
        Input positions, usually nx2
    norm_rows : bool, optional
        If True, rows will be normed to 1, by default True

    Returns
    -------
    Tensor
        Similarity Matrix
    """

    assert isinstance(positions, Tensor)

    similarity_matrix = 1-cdist(positions,positions, p=2)
    if norm_rows:
        similarity_matrix = row_norm(similarity_matrix)
    similarity_matrix.requires_grad=positions.requires_grad

    return similarity_matrix


def embedding_second_order_proximity(
    positions: Tensor,
    norm_rows: bool = True,
) -> Tensor:
    """
    Derives first the pairwise distances, then compares these distance vectors
    between positions to derive proximities of second order.

    Parameters
    ----------
    positions : Union[Tensor,ndarray]
        Input positions, usually nx2
    norm_rows : bool, optional
        If True, rows will be normed to 1, by default True

    Returns
    -------
    Tensor
        Similarity Matrix
    """
    similarity_matrix = embedding_first_order_proximity(positions, norm_rows=False)
    similarity_matrix = matrix_cosine(similarity_matrix)

    if norm_rows:
        similarity_matrix = row_norm(similarity_matrix)

    similarity_matrix.requires_grad=positions.requires_grad

    return similarity_matrix


def matrix_cosine(mat:Tensor)->Tensor:
    """
    Applies cosine similarity to a matrix, since pdist or cdist in PyTorch
    only computes Minkowski metrices.

    Parameters
    ----------
    mat : Tensor
        NxN Input

    Returns
    -------
    Tensor
        Cosine Similarity Matrix
    """
    return cosine_similarity(mat[..., None, :, :], mat[..., :, None, :], dim=-1).fill_diagonal_(0)
