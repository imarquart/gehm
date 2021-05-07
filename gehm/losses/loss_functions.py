from gehm.utils.distances import (
    embedding_first_order_proximity,
    embedding_second_order_proximity,
)
from torch import Tensor
import torch
import logging
from typing import Union

from torch.nn import CrossEntropyLoss, KLDivLoss


def first_deg_loss(
    similarity_matrix: Tensor,
    positions: Tensor,
    norm_rows: bool = True,
    beta: Union[float, int] = 2,
) -> Tensor:
    """
    First order proximity KL Divergence between similarity matrix and embedded positions.
    Includes beta parameter to emphasize non-zero elements in similarity matrix.

    Parameters
    ----------
    similarity_matrix : Tensor
        Similarity matrix of dimension NxN
    positions : Tensor
        Positions of dimension NxE where E is dimension of embedded space
    norm_rows : bool, optional
        Whether to norm rows of embedded proximities to 1, by default True
    beta : Union[float,int], optional
        Weight given to non-zero elements in similarity matrix
        Set larger than 1 to give higher weights to similarity ties that exist, by default 2

    Returns
    -------
    Tensor
        batch-mean KL Divergence

    """

    if not similarity_matrix.shape[-2] == positions.shape[-2]:
        msg = "Batch size mismatch: Second to last dimension (observations) differ. Similarities: {}, Positions {}".format(
            similarity_matrix.shape[-2], positions.shape[-2]
        )
        logging.error(msg)
        raise ValueError(msg)

    if not isinstance(positions, Tensor):
        msg = "Warning: loss function input should be Tensor, is {}".format(
            type(positions)
        )
        logging.warning(msg)
        print(msg)  # TODO Remove
        positions = torch.as_tensor(positions)



    embedding_similarity = embedding_first_order_proximity(positions=positions, norm_rows=norm_rows)

    if not embedding_similarity.shape == similarity_matrix.shape:
        msg = "Loss function size mismatch. After embedding calculations, Label similarities: {}, Embedded similarities {}".format(
            similarity_matrix.shape, positions.shape
        )
        logging.error(msg)
        raise ValueError(msg)

    zero_mask = similarity_matrix == 0
    weights = torch.ones(zero_mask.shape) * beta
    weights[zero_mask] = 1
    loss = KLDivLoss(reduction="none")
    KLloss = loss(similarity_matrix, embedding_similarity) * weights

    return KLloss.sum(-1).mean()

def second_deg_loss(
    similarity_matrix: Tensor,
    positions: Tensor,
    norm_rows: bool = True,
    beta: Union[float, int] = 2,
) -> Tensor:
    """
    Second order proximity KL Divergence between similarity matrix and embedded positions.
    Includes beta parameter to emphasize non-zero elements in similarity matrix.

    Parameters
    ----------
    similarity_matrix : Tensor
        Similarity matrix of dimension NxN
    positions : Tensor
        Positions of dimension NxE where E is dimension of embedded space
    norm_rows : bool, optional
        Whether to norm rows of embedded proximities to 1, by default True
    beta : Union[float,int], optional
        Weight given to non-zero elements in similarity matrix
        Set larger than 1 to give higher weights to similarity ties that exist, by default 2

    Returns
    -------
    Tensor
        batch-mean KL Divergence

    """

    if not similarity_matrix.shape[-2] == positions.shape[-2]:
        msg = "Batch size mismatch: Second to last dimension (observations) differ. Similarities: {}, Positions {}".format(
            similarity_matrix.shape[-2], positions.shape[-2]
        )
        logging.error(msg)
        raise ValueError(msg)

    if not isinstance(positions, Tensor):
        msg = "Warning: loss function input should be Tensor, is {}".format(
            type(positions)
        )
        logging.warning(msg)
        print(msg)  # TODO Remove
        positions = torch.as_tensor(positions)



    embedding_similarity = embedding_second_order_proximity(positions=positions, norm_rows=norm_rows)

    if not embedding_similarity.shape == similarity_matrix.shape:
        msg = "Loss function size mismatch. After embedding calculations, Label similarities: {}, Embedded similarities {}".format(
            similarity_matrix.shape, positions.shape
        )
        logging.error(msg)
        raise ValueError(msg)

    zero_mask = similarity_matrix == 0
    weights = torch.ones(zero_mask.shape) * beta
    weights[zero_mask] = 1
    loss = KLDivLoss(reduction="none")
    KLloss = loss(similarity_matrix, embedding_similarity) * weights

    return KLloss.sum(-1).mean()
