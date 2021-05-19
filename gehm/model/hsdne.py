import torch
from gehm.model.positions import Circle, Disk, Disk2, Position, SDNEPosition
from gehm.model.distances import matrix_cosine
from gehm.utils.funcs import row_norm
from gehm.model.layers.hLinear import hLinear
from typing import Union, Tuple


class hSDNEmodel(torch.nn.Module):
    def __init__(self, dim_input:int, dim_intermediate:int, dim_embedding:int, activation:torch.nn.Module=torch.nn.Tanh,
    nr_encoders:int=3, nr_decoders:int=3):
        super(hSDNEmodel, self).__init__()

        # Initialize Position Module
        #self.position=SDNEPosition(dim_embedding)
        self.position2=Disk2(max_value=0.9)
        self.position=Circle()

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
        LayerList.append(hLinear(self.dim_input,self.dim_intermediate, activation=activation))
        for i in range(0,nr_encoders-2):
            LayerList.append(hLinear(self.dim_intermediate,self.dim_intermediate, activation=activation))
        
        # Build sequential network
        self.encoder=torch.nn.ModuleList(LayerList)

        self.pos_code1=hLinear(self.dim_intermediate,1, activation=activation)
        self.pos_code2=hLinear(self.dim_intermediate,2, activation=activation)
        # Init layers
        LayerList=[]
        LayerList.append(hLinear(self.dim_embedding,self.dim_intermediate, activation=activation))
        for i in range(0,nr_encoders-2):
            LayerList.append(hLinear(self.dim_intermediate,self.dim_intermediate, activation=activation))
        LayerList.append(hLinear(self.dim_intermediate,self.dim_input, activation=activation))
        # Build sequential network
        self.decoder=torch.nn.ModuleList(LayerList)


    def forward(self, batch:torch.Tensor, attn:torch.Tensor, hierarchy:torch.tensor)->Tuple[torch.Tensor,torch.Tensor]:
        output=batch
        for i, l in enumerate(self.encoder):
            output=l(output,attn)
        position=torch.zeros(hierarchy.shape[0], self.dim_embedding)
        # TODO Generalize here
        position1=self.pos_code1(output[hierarchy==0,:],attn)
        position2=self.pos_code2(output[hierarchy==1,:],attn)
        position1 = self.position(position1)
        position2 = self.position2(position2)
        position[hierarchy==0,:]=position1
        position[hierarchy==1,:]=position2
        output2 = position.clone()
        for i, l in enumerate(self.decoder):
            output2=l(output2,attn)
        return position, output2