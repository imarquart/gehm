os.chdir("../")

from numpy import ndarray
from torch import tensor
from typing import Union, Optional
import torch
import numpy
from torch import int16

from gehm.utils.funcs import rescale_position
from gehm.utils.position import position, circle

p1=position(3,2,requires_grad=False,dtype=torch.int8)
asdf=numpy.ones([10,3])
asdf2=p1.embed(asdf)
print(asdf)
print(asdf2)

c1=circle(requires_grad=False)
asdf=numpy.ones([10,1])
asdf2=c1.embed(asdf)
print(asdf)
print(asdf2)
