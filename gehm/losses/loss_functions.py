from gehm.model.distances import (
    embedding_first_order_proximity,
    embedding_second_order_proximity,
)
from torch import Tensor
import torch
import logging
from typing import Union

from torch.nn import KLDivLoss


class WeightedLoss(torch.nn.Module):
    def __init__(self, embedding_weight:float, first_deg_weight: float, beta: Union[float, int]=2, norm_rows: bool = True, device: str = "cpu"):
        """Loss function weighting KL divergence for both first-degree proximity and second-degree proximity

        Parameters
        ----------
        first_deg_weight : float
            Weight of first-degree proximity divergence
        norm_rows : bool, optional
            Whether to norm rows of embedded proximities to 1, by default True
        beta : Union[float,int], optional
            Weight given to non-zero elements in similarity matrix
            Set larger than 1 to give higher weights to similarity ties that exist, by default 2
        """
        super(WeightedLoss, self).__init__()
        self.weight=torch.tensor(first_deg_weight).to(device)
        self.emb=torch.tensor(embedding_weight).to(device)
        self.firstdegloss=FirstDegLoss(norm_rows, beta=1, device=device)
        self.seconddegloss=SecondDegLoss(norm_rows, beta=1, device=device)
        self.firstdegencloss=FirstDegLossEncoder(norm_rows, beta, device=device)
        self.mindloss=MinDistLoss(device=device)
        

    def forward(self, similarity: torch.Tensor, similarity2: torch.Tensor,positions: torch.Tensor, est_similarity: torch.Tensor = None)->torch.Tensor:
        """
        Parameters
        ----------
        similarity_matrix : Tensor
            Similarity matrix of dimension NxN
        positions : Tensor
            Positions of dimension NxE where E is dimension of embedded space

        Returns
        -------
        Tensor
            batch-mean KL Divergence

        """
        if est_similarity is None:
            est_similarity=similarity

        first_loss=self.firstdegloss(positions, similarity)
        second_loss=self.seconddegloss(positions, similarity2)
        embedded_loss=self.weight*first_loss + (1-self.weight)*second_loss
        decoder_loss=self.firstdegencloss(est_similarity, similarity)
        
        mixed_loss= self.emb*embedded_loss+ (1-self.emb)*decoder_loss
        return mixed_loss, first_loss, second_loss, decoder_loss


class MinDistLoss(torch.nn.Module):
    def __init__(self, norm_rows: bool = True, min_distance:float=0.1, device: str = "cpu"):
        """
        First order proximity KL Divergence between similarity matrix and embedded positions.
    Includes beta parameter to emphasize non-zero elements in similarity matrix.

        Parameters
        ----------
        norm_rows : bool, optional
            Whether to norm rows of embedded proximities to 1, by default True
        beta : Union[float,int], optional
            Weight given to non-zero elements in similarity matrix
            Set larger than 1 to give higher weights to similarity ties that exist, by default 2
        """
        super(MinDistLoss, self).__init__()
        self.norm_rows = torch.tensor(norm_rows).to(device)
        self.min_distance = torch.tensor(min_distance).to(device)
        

    def forward(self, positions):
        """
        First order proximity KL Divergence between similarity matrix and embedded positions.
        Includes beta parameter to emphasize non-zero elements in similarity matrix.

        Parameters
        ----------
        positions : Tensor
            Positions of dimension NxE where E is dimension of embedded space

        Returns
        -------
        Tensor
            batch-mean KL Divergence

        """


        embedding_similarity = embedding_first_order_proximity(
            positions=positions, norm_rows=self.norm_rows
        )

        ldp=torch.nn.ReLU()
        md_loss=torch.norm(ldp(self.min_distance - embedding_similarity),p=2,dim=-1).sum()

        return md_loss




class FirstDegLoss(torch.nn.Module):
    def __init__(self, norm_rows: bool = True, beta: Union[float, int] = 2, device: str = "cpu"):
        """
        First order proximity KL Divergence between similarity matrix and embedded positions.
    Includes beta parameter to emphasize non-zero elements in similarity matrix.

        Parameters
        ----------
        norm_rows : bool, optional
            Whether to norm rows of embedded proximities to 1, by default True
        beta : Union[float,int], optional
            Weight given to non-zero elements in similarity matrix
            Set larger than 1 to give higher weights to similarity ties that exist, by default 2
        """
        super(FirstDegLoss, self).__init__()
        self.norm_rows = torch.tensor(norm_rows).to(device)
        self.beta = torch.tensor(beta).to(device)
        

    def forward(self, positions, similarity):
        """
        First order proximity KL Divergence between similarity matrix and embedded positions.
        Includes beta parameter to emphasize non-zero elements in similarity matrix.

        Parameters
        ----------
        similarity : Tensor
            Similarity matrix of dimension NxN
        positions : Tensor
            Positions of dimension NxE where E is dimension of embedded space

        Returns
        -------
        Tensor
            batch-mean KL Divergence

        """
        if not similarity.shape[-2] == positions.shape[-2]:
            msg = "Batch size mismatch: Second to last dimension (observations) differ. Similarities: {}, Positions {}".format(
                similarity.shape[-2], positions.shape[-2]
            )
            logging.error(msg)
            raise ValueError(msg)

        embedding_similarity = embedding_first_order_proximity(
            positions=positions, norm_rows=self.norm_rows
        ).to(similarity.device)

        if not embedding_similarity.shape == similarity.shape:
            msg = "Loss function size mismatch. After embedding calculations, original similarities: {}, Embedded similarities {}".format(
                similarity.shape, embedding_similarity.shape
            )
            logging.error(msg)
            raise ValueError(msg)

        zero_mask_sim = similarity == 0
        weights = torch.ones(zero_mask_sim.shape).to(similarity.device) * self.beta
        weights[zero_mask_sim] = 1


        loss = torch.norm((embedding_similarity-similarity)*weights,p=2,dim=-1).to(similarity.device).mean()/self.beta

        return loss



class SecondDegLoss(torch.nn.Module):
    def __init__(self, norm_rows: bool = True, beta: Union[float, int] = 2, device: str = "cpu"):
        """
        First order proximity KL Divergence between similarity matrix and embedded positions.
        Includes beta parameter to emphasize non-zero elements in similarity matrix.

        Parameters
        ----------
        norm_rows : bool, optional
            Whether to norm rows of embedded proximities to 1, by default True
        beta : Union[float,int], optional
            Weight given to non-zero elements in similarity matrix
            Set larger than 1 to give higher weights to similarity ties that exist, by default 2
        """
        super(SecondDegLoss, self).__init__()
        self.norm_rows = torch.tensor(norm_rows).to(device)
        self.beta = torch.tensor(beta).to(device)
        

    def forward(self, positions, similarity):
        """
        First order proximity KL Divergence between similarity matrix and embedded positions.
        Includes beta parameter to emphasize non-zero elements in similarity matrix.

        Parameters
        ----------
        similarity : Tensor
            Similarity matrix of dimension NxN
        positions : Tensor
            Positions of dimension NxE where E is dimension of embedded space

        Returns
        -------
        Tensor
            batch-mean KL Divergence

        """
        if not similarity.shape[-2] == positions.shape[-2]:
            msg = "Batch size mismatch: Second to last dimension (observations) differ. Similarities: {}, Positions {}".format(
                similarity.shape[-2], positions.shape[-2]
            )
            logging.error(msg)
            raise ValueError(msg)

        embedding_similarity = embedding_second_order_proximity(
            positions=positions, norm_rows=self.norm_rows
        )

        if not embedding_similarity.shape == similarity.shape:
            msg = "Loss function size mismatch. After embedding calculations, Label similarities: {}, Embedded similarities {}".format(
                similarity.shape, positions.shape
            )
            logging.error(msg)
            raise ValueError(msg)

        zero_mask = similarity == 0
        weights = torch.ones(zero_mask.shape).to(self.beta.device) * self.beta
        weights[zero_mask] = 1

        return torch.norm((embedding_similarity-similarity)*weights,p=2,dim=-1).mean()/self.beta


