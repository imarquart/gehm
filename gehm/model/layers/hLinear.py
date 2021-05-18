import torch
from typing import Union, Tuple


class hLinear(torch.nn.Module):
    def __init__(self, dim_input:int, dim_output:int, activation:torch.nn.Module=torch.nn.Tanh):
        super(hLinear, self).__init__()

        self.dim_input=dim_input
        self.dim_output=dim_output



        self.ll=torch.nn.Linear(self.dim_input,self.dim_output)
        self.act=activation()

    def forward(self, batch:torch.Tensor, attn:torch.Tensor,)->torch.Tensor:
        #print("batch_shape {}, attn_shape: {}, dim_input: {}, dim_output:{}".format(batch.shape,attn.shape,self.dim_input, self.dim_output))
        if batch.shape==attn.shape:
            batch=torch.mul(batch,attn)
        batch=self.ll(batch)
        return self.act(batch)