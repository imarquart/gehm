import torch
import numpy as np
from typing import Union


def rescale_position(input: torch.Tensor) -> torch.Tensor:
    """Rescales vector to range between 0 and 1 based on the last dimension

    Parameters
    ----------
    input : torch.Tensor
        Input

    Returns
    -------
    torch.Tensor
        Rescaled vector
    """
    range = (
        torch.max(input, -1, keepdim=True)[0] - torch.min(input, -1, keepdim=True)[0]
    )
    output = torch.true_divide(input - torch.min(input, -1, keepdim=True)[0], range)
    output[output.isnan()] = 1
    return output


def row_norm(input: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    rowsums = input.sum(axis=-1, keepdims=True)
    rowsums[rowsums == 0] = 1
    return input / rowsums

