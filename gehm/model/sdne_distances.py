import networkx as nx
from typing import Union
from torch import Tensor
import networkx as nx
import torch
from torch.nn.functional import cosine_similarity
from torch import cdist
from gehm.utils.funcs import row_norm


def sdne_prox_distance(
    positions: Tensor
) -> Tensor:
    """
    A simple application of euclidian distance to the positions vector in the embedding space.

    Parameters
    ----------
    positions : Tensor
        Input positions, usually nx2

    Returns
    -------
    Tensor
        _Distance_ (not proximity) Matrix
    """

    assert isinstance(positions, Tensor)

    return cdist(positions, positions, p=2).to(positions.device)
