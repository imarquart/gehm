import networkx as nx
from typing import Union
from torch import Tensor
import networkx as nx
import torch
from torch.nn.functional import cosine_similarity
from torch import cdist
from gehm.utils.funcs import row_norm


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

    similarity_matrix = cdist(positions, positions, p=2).to(positions.device)
    if norm_rows:
        similarity_matrix = row_norm(similarity_matrix)
    similarity_matrix = 1-similarity_matrix


    return similarity_matrix-torch.eye(similarity_matrix.shape[0]).to(positions.device)


def embedding_second_order_proximity(
    positions: Tensor, norm_rows: bool = True,
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
    similarity_matrix = embedding_first_order_proximity(positions, norm_rows=norm_rows)
    similarity_matrix = matrix_cosine(similarity_matrix)

    if norm_rows:
        similarity_matrix = row_norm(similarity_matrix)
    return similarity_matrix.to(positions.device)


def matrix_cosine(mat: Tensor) -> Tensor:
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
    return cosine_similarity(
        mat[..., None, :, :], mat[..., :, None, :], dim=-1
    ).fill_diagonal_(0).to(mat.device)
