from tests.test_data import create_test_data
from gehm.losses.loss_functions import *
from gehm.datasets.nx_datasets import nx_dataset_onelevel
import pytest
import numpy as np
import torch

@pytest.mark.losses
def test_first_deg_loss(create_test_data):

    G,G_undir= create_test_data


    dataset=nx_dataset_onelevel(G)

    nodes,sim1,sim2=dataset[0:6]

    positions=torch.as_tensor(np.random.rand(sim1.shape[0],2))

    # Test size exception
    with pytest.raises(ValueError):
        first_deg_loss(sim1,positions[0:positions.shape[0]-2,:])

    positions=torch.as_tensor(np.ones([sim1.shape[0],2]))
    assert (first_deg_loss(sim1,positions)<0), "Neg loss should be <0, is {}".format(first_deg_loss(sim1,positions))
    
@pytest.mark.losses
def test_second_deg_loss(create_test_data):

    G,G_undir= create_test_data


    dataset=nx_dataset_onelevel(G)

    nodes,sim1,sim2=dataset[0:6]

    positions=torch.as_tensor(np.random.rand(sim1.shape[0],2))

    # Test size exception
    with pytest.raises(ValueError):
        second_deg_loss(sim1,positions[0:positions.shape[0]-2,:])

    positions=torch.as_tensor(np.ones([sim1.shape[0],2]))
    assert (second_deg_loss(sim1,positions)<0), "Neg loss should be <0, is {}".format(second_deg_loss(sim1,positions))


