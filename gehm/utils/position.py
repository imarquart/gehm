from numpy import ndarray, sin, cos
from torch import tensor
from typing import Union, Optional
import torch
import numpy
from torch import int16

from gehm.utils.funcs import rescale_position


class position:
    def __init__(
        self, dim_orig: int, dim_emb: int, requires_grad: bool, dtype=torch.float64
    ):
        """
        'Abstract' class for positional embeddings.

        Parameters
        ----------
        dim_orig : int
            Dimension of the original space. Convention: Last dimension of the supplied tensor.
            All other dimensions are considered observations/batches.
        dim_emb : int
            Dimension of the embedding space.
        requires_grad : bool
            pytorch parameter applied to output
        dtype : [type], optional
            pytorch parameter cast onto output
        """

        self.dim_orig = dim_orig
        self.dim_emb = dim_emb
        self.requires_grad = requires_grad
        self.dtype = dtype

    def embed(self, input: Union[torch.Tensor, ndarray]) -> torch.Tensor:
        """
        Generates embedded position from input

        Parameters
        ----------
        input : Union[torch.Tensor, ndarray]
            Input tensor

        Returns
        -------
        torch.Tensor
            Output in embedded space
        """

        if len(input.shape)==1:
            input=input.reshape([-1,1])

        assert (
            input.shape[-1] == self.dim_orig
        ), "Input tensor's dimension is {}, position is initialized with input dimension {}".format(
            input.shape[-1], self.dim_orig
        )
        if isinstance(input, ndarray):
            input = torch.tensor(input)

        transform = self.transformation_function(input)
        transform.requires_grad = self.requires_grad
        transform = transform.to(self.dtype)
        return transform

    def transformation_function(
        self, input: Union[torch.Tensor, ndarray]
    ) -> Union[torch.Tensor, ndarray]:
        """
        Implement this function in your subclass.
        This example simply cuts the dimensions.

        Parameters
        ----------
        input : Union[torch.Tensor, ndarray]
            input

        Returns
        -------
        Union[torch.Tensor, ndarray]
            output
        """
        # Implement transformation here
        transformation = input[..., : self.dim_emb]
        return transformation


class circle(position):
    def __init__(self, requires_grad: bool = True, dtype=torch.float64, max_value:Union[int,float]=1):
        """
        Projects degrees onto a circle.
        See transformation_function() for details

        Parameters
        ----------
        requires_grad : bool, optional
            by default True
        dtype : [type], optional
            by default torch.float64
        max_value : Union[int,float], optional
            The maximum value of each input element - used for rescaling
        """
        self.dim_orig = 1
        self.dim_emb = 2
        self.max_value=max_value
        self.requires_grad = requires_grad
        self.dtype = dtype

        super().__init__(self.dim_orig, self.dim_emb, requires_grad, dtype)

    def transformation_function(
        self, input: Union[torch.Tensor, ndarray]
    ) -> Union[torch.Tensor, ndarray]:
        """
        Project positional values in the range [0,self.max_value] into a circle
        in a 2d plane. Input values are understood as degrees, scaled between 0 and self.max_value.
        e.g. degree=360*input/self.max_value

        Parameters
        ----------
        input : Union[torch.Tensor, ndarray]
            Input degrees as self.max_value*degree/360. Last dimension must be 1.

        Returns
        -------
        Union[torch.Tensor, ndarray]
            Output in (x,y) on the unit circle. Last dimension will be 2.
        """
        input=input/self.max_value
        y = sin(input * 360)
        x = cos(input * 360)
        return torch.cat([x, y], -1)


class disk(position):
    def __init__(self, requires_grad: bool = True, dtype=torch.float64, max_value=None):
        """
        Projects (x,y) coordinates into a disc inside a unit circle.
        See transformation_function() for details

        Parameters
        ----------
        requires_grad : bool, optional
            by default True
        dtype : [type], optional
            by default torch.float64
        """
        self.dim_orig = 2
        self.dim_emb = 2
        self.requires_grad = requires_grad
        self.dtype = dtype
        self.max_value=max_value

        super().__init__(self.dim_orig, self.dim_emb, requires_grad, dtype)

    def transformation_function(
        self, input: Union[torch.Tensor, ndarray]
    ) -> Union[torch.Tensor, ndarray]:
        """
        Project positional values into the a disk of radius self.max_value
        For example, if max_value is 1, values get projected into the unit disk.
        Here, for coordinates z with ||z||>1, the coordinate are set to the unit circle z/||z||

        Parameters
        ----------
        input : Union[torch.Tensor, ndarray]
            Input coordinates. Last dimension must be 2.

        Returns
        -------
        Union[torch.Tensor, ndarray]
            Output in (x,y) on the unit disk. Last dimension will be 2.
        """

        # Coordinates outside the unit circle will be normed to length 1
        inorm=torch.norm(input,dim=-1, keepdim=True,p=2).squeeze()
        if (inorm>self.max_value).any():
            input[inorm>1,...]=torch.div(input[inorm>1,...],inorm[inorm>1])*self.max_value

        return input