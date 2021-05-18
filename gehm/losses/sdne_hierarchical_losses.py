from gehm.model.sdne_distances import sdne_prox_distance
from torch import Tensor
import torch
import logging
from typing import Union

class hSDNESELoss(torch.nn.Module):
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
        super(hSDNESELoss, self).__init__()
        self.beta = torch.tensor(beta).to(device)
        self.beta.requires_grad=False
        

    def forward(self, est_similarity, similarity, attn):
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
        weights=weights*attn
        weights.requires_grad=False

        loss = torch.norm((est_similarity-similarity)*weights,p=2,dim=-1).mean()

        return loss


class hSDNEVarianceLoss(torch.nn.Module):
    def __init__(self, device:str="cpu", max_var:int=1):
        """
        In-group variance incentive
        """
        super(hSDNEVarianceLoss, self).__init__()

        self.max_var=torch.tensor(max_var).to(device)
        self.max_var.requires_grad=False

      
    def forward(self, positions:torch.Tensor, hierarchy:torch.Tensor):          
        var_loss=self.max_var-torch.var(positions)
    
        return var_loss

class hSDNEAreaLoss(torch.nn.Module):
    def __init__(self, device:str="cpu", max_area:int=4):
        """
        Area bias
        """
        super(hSDNEAreaLoss, self).__init__()

        self.max_area=torch.tensor(max_area).to(device)
        self.max_area.requires_grad=False

    def forward(self, positions:torch.Tensor):        
        
        pos_max=torch.max(positions, axis=0)[0]
        pos_min=torch.min(positions, axis=0)[0]    

        vol=torch.abs(pos_max-pos_min)
        vol=vol[0]*vol[1]

        vol_loss = self.max_area-vol   
    
        return vol_loss

class hSDNEProximityLoss(torch.nn.Module):
    def __init__(self, device:str="cpu"):
        """
        First order proximity between similarity matrix and embedded positions.
        Includes beta parameter to emphasize non-zero elements in similarity matrix.
        """
        super(hSDNEProximityLoss, self).__init__()
        

    def forward(self, positions:torch.Tensor, similarity:torch.Tensor, indecies:torch.Tensor, attn:torch.Tensor):
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

        #

        cut_sim=1-similarity[:,indecies] # Creates a new tensor
        cut_attn=attn[:,indecies]
        distances=torch.mul(distances,cut_attn)
        cut_sim=torch.mul(cut_sim,cut_attn)
        #loss = torch.mul(distances,cut_sim).sum().to(positions.device)
        #loss2 = torch.abs(torch.mul(1-distances,1-cut_sim)).sum().to(positions.device)

        loss = torch.abs(torch.mul(1-cut_sim,torch.square(distances-cut_sim))).sum().to(positions.device)

        return loss