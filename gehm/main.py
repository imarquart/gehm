import os

os.chdir("../")

from numpy import ndarray, cos, sin
from torch import tensor
from typing import Union, Optional
import torch
import numpy as np
from torch import int16
from tests.test_data import create_test_data
import networkx as nx
from gehm.model.positions import position, circle, disk
from gehm.losses.loss_functions import *
from gehm.datasets.nx_datasets import *


G,G_undir= create_test_data()


dataset=nx_dataset_onelevel(G)
first_deg_loss=FirstDegLoss()

nodes,sim1,sim2=dataset[0:6]

positions=torch.as_tensor(np.random.rand(sim1.shape[0],2))

first_deg_loss(positions,sim1)

 # Test 1: Scaling between 0 and 1
c1=circle(max_value=1)
test_input=np.zeros([3,1])
test_input[0,:]=0
test_input[1,:]=0.5
test_input[2,:]=0.25
test_input=torch.tensor([test_input,test_input,test_input])
output=c1(test_input)

p1=position(3,2)
asdf=torch.as_tensor(numpy.random.rand(10,3))
print(asdf)
asdf2=p1(asdf)
print(asdf2)

c1=circle()
asdf=torch.as_tensor(numpy.random.rand(10,1))
print(asdf)
asdf2=c1(asdf)
print(asdf2)

d1=disk()
asdf=torch.as_tensor(numpy.random.rand(10,2))*1000
print(asdf)
asdf2=d1(asdf)
print(asdf2)
