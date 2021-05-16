import torch
from gehm.losses.loss_functions import FirstDegLoss, SecondDegLoss, WeightedLoss
from gehm.model.positions import Circle, Disk, Disk2, Position, SDNEPosition
from gehm.model.distances import matrix_cosine
from gehm.utils.funcs import row_norm
from typing import Union, Tuple


class tSDNEmodel(torch.nn.Module):
    def __init__(self, dim_input:int, dim_intermediate:int, dim_embedding:int, activation:torch.nn.Module=torch.nn.Tanh,
                 nr_encoders:int=3, nr_decoders:int=3, nr_heads:int=6, dropout:float=0.1, encoder_activation:str="relu"):
        super(tSDNEmodel, self).__init__()

        # Initialize Position Module
        self.position=SDNEPosition(dim_embedding)
        #self.position=Disk2()

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

        # Init Encoder
        LayerList=[]
        encoder_layer=torch.nn.TransformerEncoderLayer(dim_input, nr_heads, dim_feedforward=dim_intermediate, dropout=dropout, activation=encoder_activation)
        #encoder=torch.nn.TransformerEncoder(encoder_layer, nr_encoders, norm=torch.nn.LayerNorm(self.dim_input))
        encoder=torch.nn.TransformerEncoder(encoder_layer, nr_encoders)
        LayerList.append(encoder)
        LayerList.append(torch.nn.Linear(self.dim_input,self.dim_embedding))
        LayerList.append(activation())
        # Build sequential network
        self.encoder=torch.nn.Sequential(*LayerList)


        LayerList2=[]
        LayerList2.append(torch.nn.Linear(self.dim_embedding,self.dim_input))
        LayerList2.append(activation())
        self.upcode=torch.nn.Sequential(*LayerList2)
        LayerList2=[]
        for i in range(0,nr_decoders+1):
            LayerList2.append(torch.nn.TransformerDecoderLayer(dim_input, nr_heads, dim_feedforward=dim_intermediate, dropout=dropout, activation=encoder_activation))
        #LayerList2.append(torch.nn.Linear(self.dim_input,self.dim_input))
        #LayerList2.append(activation())
        # Build sequential network
        #decoder_layer=torch.nn.TransformerDecoderLayer(dim_input, nr_heads, dim_feedforward=dim_intermediate, dropout=dropout, activation=encoder_activation)
        #self.decoder=torch.nn.TransformerDecoder(decoder_layer, nr_encoders, norm=torch.nn.LayerNorm(self.dim_input))#torch.nn.Sequential(*LayerList2)
        decoder_layer=torch.nn.TransformerEncoderLayer(dim_input, nr_heads, dim_feedforward=dim_intermediate, dropout=dropout, activation=encoder_activation)
        #self.decoder=torch.nn.TransformerEncoder(decoder_layer, nr_encoders, norm=torch.nn.LayerNorm(self.dim_input))
        self.decoder=torch.nn.TransformerEncoder(decoder_layer, nr_encoders)

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
        #self.decoder=torch.nn.Sequential(*LayerList)

    def forward(self, batch:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:

        output=self.encoder(batch.unsqueeze(0))
        position = self.position(output)
        output2 = position.clone()
        output2 =self.upcode(output2)
        output2= self.decoder(output2)
        return position.squeeze(0), output2.squeeze(0)


class SDNEmodel(torch.nn.Module):
    def __init__(self, dim_input:int, dim_intermediate:int, dim_embedding:int, activation:torch.nn.Module=torch.nn.Tanh,
    nr_encoders:int=3, nr_decoders:int=3):
        super(SDNEmodel, self).__init__()

        # Initialize Position Module
        self.position=SDNEPosition(dim_embedding)
        #self.position=Disk2()

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
       


    def forward(self, batch:torch.Tensor)->Tuple[torch.Tensor,torch.Tensor]:

        output=self.encoder(batch)
        position = self.position(output)
        output2 = position.clone()
        output2= self.decoder(output2)
        return position, output2