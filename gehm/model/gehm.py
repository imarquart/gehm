import torch
from gehm.losses.loss_functions import FirstDegLoss, SecondDegLoss, WeightedLoss
from gehm.model.positions import Circle, Disk, Position
from gehm.utils.distances import matrix_cosine
from gehm.utils.funcs import row_norm
from typing import Union

class OneLevelGehm(torch.nn.Module):
    def __init__(self, dim_input:int, dim_intermediate:int , position: Position, activation:torch.nn.Module=torch.nn.Tanh,nr_transformers:int=3,nr_heads:int=6,nr_decoder:int=3):
        super(OneLevelGehm, self).__init__()

        # Initialize Position Module
        self.position=position()
        #self.position=Disk()
        #self.position=Position(2,2)
        # Init dimensions
        self.dim_input=dim_input
        self.dim_output=self.position.dim_orig
        self.dim_embedding=self.position.dim_emb
        self.dim_intermediate=dim_intermediate
        self.nr_decoder=nr_decoder
        self.nr_transformers=nr_transformers

        # Init layers
        LayerList=[]
        for i in range(0,nr_transformers+1):
            LayerList.append(torch.nn.TransformerEncoderLayer(dim_input, nr_heads, dim_feedforward=dim_intermediate, dropout=0.1, activation='relu'))


        # Build sequential network
        self.net1=torch.nn.Sequential(*LayerList)


        if self.nr_decoder < 3:
            self.nr_decoder=3
        LayerList2=[]
        
        red=self.dim_input//self.nr_decoder
        redlist=list(range(dim_input, 0, -red))
        LayerList2.append(torch.nn.Linear(dim_input,redlist[0]))
        LayerList2.append(activation())
        for i in range(1,self.nr_decoder):
            LayerList2.append(torch.nn.Linear(redlist[i-1],redlist[i]))
            LayerList2.append(activation())

        LayerList2=[]
        LayerList2.append(torch.nn.Linear(dim_input,self.dim_output))
        LayerList2.append(activation())
        LayerList2.append(self.position)
        self.net2=torch.nn.Sequential(*LayerList2)

        


    def forward(self, batch:torch.Tensor)->torch.Tensor:

        output=self.net1(batch.unsqueeze(0))
        output = self.net2(output)
        return output.squeeze(0) #self.position(output)



class OneLevelGehm2(torch.nn.Module):
    def __init__(self, dim_input:int, dim_intermediate:int , position: Position, activation:torch.nn.Module=torch.nn.Tanh,nr_transformers:int=3,nr_heads:int=6,nr_decoder:int=3):
        super(OneLevelGehm2, self).__init__()

        # Initialize Position Module
        self.position=position()
        #self.position=Disk()
        #self.position=Position(2,2)
        # Init dimensions
        self.dim_input=dim_input
        self.dim_output=self.position.dim_orig
        self.dim_embedding=self.position.dim_emb
        self.dim_intermediate=dim_intermediate
        self.nr_decoder=nr_decoder
        self.nr_transformers=nr_transformers

        # Init layers
        LayerList=[]
        for i in range(0,nr_transformers+1):
            LayerList.append(torch.nn.TransformerEncoderLayer(dim_input, nr_heads, dim_feedforward=dim_intermediate, dropout=0.1, activation='relu'))


        # Build sequential network
        self.net1=torch.nn.Sequential(*LayerList)
        
        LayerList2=[]
        LayerList2.append(torch.nn.Linear(dim_input,self.dim_output))
        LayerList2.append(activation())
        LayerList2.append(self.position)
        self.net2=torch.nn.Sequential(*LayerList2)

        LayerList3=[]
        LayerList3.append(torch.nn.Linear(self.dim_output,dim_input))
        LayerList3.append(activation())
        for i in range(0,nr_transformers+1):
            LayerList.append(torch.nn.TransformerDecoderLayer(dim_input, nr_heads, dim_feedforward=dim_intermediate, dropout=0.1, activation='relu'))
        self.net3=torch.nn.Sequential(*LayerList3)

       


    def forward(self, batch:torch.Tensor)->torch.Tensor:

        output=self.net1(batch.unsqueeze(0))
        position = self.net2(output)
        output2 = position.clone()
        output2= self.net3(output2)
        return position.squeeze(0), output2.squeeze(0)  #self.position(output)