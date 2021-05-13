from gehm.model.sdne_distances import sdne_prox_distance
from torch import Tensor
import torch
import logging
from typing import Union

from torch.nn import KLDivLoss



class SDNESELoss(torch.nn.Module):
    def __init__(self, beta: Union[float, int] = 2, device: str = "cpu"):
        """
        "Second order proximity" between similarity matrix and estimated similarity.
        Includes beta parameter to emphasize non-zero elements in similarity matrix.

        Parameters
        ----------
        beta : Union[float,int], optional
            Weight given to non-zero elements in similarity matrix
            Set larger than 1 to give higher weights to similarity ties that exist, by default 2
        """
        super(SDNESELoss, self).__init__()
        self.beta = torch.tensor(beta).to(device)
        

    def forward(self, est_similarity, similarity):
        """
        Second order proximity between similarity matrix and estimated similarity.
        Includes beta parameter to emphasize non-zero elements in similarity matrix.

        Parameters
        est_similarity
        similarity : Tensor
            Estimated Similarity Matrix of dimension BxN
        similarity : Tensor
            Ground Truth Similarity Matrix of dimension NxN
        indecies : Tensor
            Indecies denoting the B selected elements

        Returns
        -------
        Tensor
            batch-mean L2 Difference

        """

        zero_mask_sim = similarity == 0
        weights = torch.ones(zero_mask_sim.shape).to(self.beta.device) * self.beta
        weights[zero_mask_sim] = 1

        loss = torch.norm((est_similarity-similarity)*weights,p=2,dim=-1).mean()

        return loss


class SDNEProximityLoss(torch.nn.Module):
    def __init__(self, device:str="cpu"):
        """
        First order proximity between similarity matrix and embedded positions.
        Includes beta parameter to emphasize non-zero elements in similarity matrix.

        Parameters
        ----------
        norm_rows : bool, optional
            Whether to norm rows of embedded proximities to 1, by default True
        beta : Union[float,int], optional
            Weight given to non-zero elements in similarity matrix
            Set larger than 1 to give higher weights to similarity ties that exist, by default 2
        """
        super(SDNEProximityLoss, self).__init__()
        

    def forward(self, positions:torch.Tensor, similarity:torch.Tensor, indecies:torch.Tensor):
        """
        First order proximity KL Divergence between similarity matrix and embedded positions.
        Includes beta parameter to emphasize non-zero elements in similarity matrix.

        Parameters
        positions : Tensor
            Embedded Positions of size NxK
        similarity : Tensor
            Ground Truth Similarity Matrix of dimension NxN

        Returns
        -------
        Tensor
            batch-mean L2 Difference

        """

        distances=sdne_prox_distance(positions)
        cut_sim=similarity[:,indecies] # Creates a new tensor

        loss = torch.mul(distances,cut_sim).sum().to(positions.device)

        return loss