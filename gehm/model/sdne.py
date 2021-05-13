import torch
from gehm.losses.loss_functions import FirstDegLoss, SecondDegLoss, WeightedLoss
from gehm.model.positions import Circle, Disk, Disk2, Position, SDNEPosition
from gehm.model.distances import matrix_cosine
from gehm.utils.funcs import row_norm
from typing import Union


class SDNEmodel(torch.nn.Module):
    def __init__(self, dim_input:int, dim_intermediate:int, dim_embedding:int, activation:torch.nn.Module=torch.nn.Tanh,
    nr_encoders:int=3, nr_decoders:int=3):
        super(SDNEmodel, self).__init__()

        # Initialize Position Module
        #self.position=SDNEPosition(dim_embedding)
        self.position=Disk2()

        # Init dimensions
        self.dim_input=dim_input
        self.dim_embedding=dim_embedding
        self.dim_intermediate=dim_intermediate

        if nr_encoders < 2:
            nr_encoders = 2
        if nr_decoders < 2:
            nr_decoders = 2
        self.nr_encoders=nr_encoders
        self.nr_decoders=nr_encoders

        # Init layers
        LayerList=[]
        LayerList.append(torch.nn.Linear(self.dim_input,self.dim_intermediate))
        LayerList.append(activation())
        for i in range(0,nr_encoders-2):
            LayerList.append(torch.nn.Linear(self.dim_intermediate,self.dim_intermediate))
            LayerList.append(activation())
        LayerList.append(torch.nn.Linear(self.dim_intermediate,self.dim_embedding))
        LayerList.append(activation())
        # Build sequential network
        self.encoder=torch.nn.Sequential(*LayerList)
     
        # Init layers
        LayerList=[]
        LayerList.append(torch.nn.Linear(self.dim_embedding,self.dim_intermediate))
        LayerList.append(activation())
        for i in range(0,nr_decoders-2):
            LayerList.append(torch.nn.Linear(self.dim_intermediate,self.dim_intermediate))
            LayerList.append(activation())
        LayerList.append(torch.nn.Linear(self.dim_intermediate,self.dim_input))
        LayerList.append(activation())
        # Build sequential network
        self.decoder=torch.nn.Sequential(*LayerList)
       


    def forward(self, batch:torch.Tensor)->torch.Tensor:

        output=self.encoder(batch)
        position = self.position(output)
        output2 = position.clone()
        output2= self.decoder(output2)
        return position, output2