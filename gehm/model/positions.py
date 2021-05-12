from typing import Union, Optional
import torch
from torch.nn.functional import normalize


class Position(torch.nn.Module):
    def __init__(self, dim_orig: int, dim_emb: int):
        """
        'Abstract' module for positional embeddings.

        Parameters
        ----------
        dim_orig : int
            Dimension of the original space. Convention: Last dimension of the supplied tensor.
            All other dimensions are considered observations/batches.
        dim_emb : int
            Dimension of the embedding space.
        """
        super(Position, self).__init__()
        self.dim_orig = dim_orig
        self.dim_emb = dim_emb
        

    def forward(self, x):
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

        if len(x.shape) == 1:
            x = x.unsqueeze(1)

        assert (
            x.shape[-1] == self.dim_orig
        ), "Input tensor's dimension is {}, position is initialized with input dimension {}".format(
            x.shape[-1], self.dim_orig
        )

        return self.transformation_function(x)

    def transformation_function(self, input: torch.Tensor) -> torch.Tensor:
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
        transformation = input.clone()
        return transformation[..., : self.dim_emb]


class Circle(Position):
    def __init__(self, max_value: Union[int, float] = 1):
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
        self.max_value = max_value
        super(Circle, self).__init__(self.dim_orig, self.dim_emb)
        

    def transformation_function(self, input: torch.Tensor) -> torch.Tensor:
        """
        Project positional values in the range [0,self.max_value] into a circle
        in a 2d plane. Input values are understood as degrees, scaled between 0 and self.max_value.
        e.g. degree=360*input/self.max_value

        Parameters
        ----------
        input : torch.Tensor
            Input degrees as self.max_value*degree/360. Last dimension must be 1.

        Returns
        -------
        torch.Tensor
            Output in (x,y) on the unit circle. Last dimension will be 2.
        """
        input = input / self.max_value
        y = torch.sin(input * 360)
        x = torch.cos(input * 360)
        return torch.cat([x, y], -1)


class Disk(Position):
    def __init__(self, max_value=None):
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
        if max_value is None:
            max_value = 1
        self.max_value = max_value
        super(Disk, self).__init__(self.dim_orig, self.dim_emb)
        

    def transformation_function(self, input: torch.Tensor) -> torch.Tensor:
        """
        Project positional values into the a disk of radius self.max_value
        For example, if max_value is 1, values get projected into the unit disk.
        Here, for coordinates z with ||z||>1, the coordinate are set to the unit circle z/||z||

        Parameters
        ----------
        input : torch.Tensor
            Input coordinates. Last dimension must be 2.

        Returns
        -------
        torch.Tensor
            Output in (x,y) on the unit disk. Last dimension will be 2.
        """

        # Coordinates outside the unit circle will be normed to length 1
        output=input.clone()
        inorm = torch.norm(input, dim=-1, keepdim=True, p=2).squeeze()
        if (inorm > self.max_value).any():
            output[...,inorm > self.max_value,:] = (
                normalize(output[...,inorm > self.max_value,:], p=2, dim=-1)
                * self.max_value
            )

        return output




class Disk2(Position):
    def __init__(self, max_value=None):
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
        if max_value is None:
            max_value = 1
        self.max_value = max_value
        super(Disk2, self).__init__(self.dim_orig, self.dim_emb)
        

    def transformation_function(self, input: torch.Tensor) -> torch.Tensor:
        """
        Project positional values into the a disk of radius self.max_value
        For example, if max_value is 1, values get projected into the unit disk.
        Here, for coordinates z with ||z||>1, the coordinate are set to the unit circle z/||z||

        Parameters
        ----------
        input : torch.Tensor
            Input coordinates. Last dimension must be 2.

        Returns
        -------
        torch.Tensor
            Output in (x,y) on the unit disk. Last dimension will be 2.
        """

        # Coordinates outside the unit circle will be normed to length 1
        output=input.clone()
        inorm = torch.norm(input, dim=-1, keepdim=True, p=2).squeeze()
        maxnorm= torch.max(inorm)#*(1+inorm.mean()/torch.max(inorm))
        #maxnorm= torch.max(inorm)*(1+inorm/torch.max(inorm))
        output=output/maxnorm.unsqueeze(-1)

        return output
