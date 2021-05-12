from tests.test_data import create_test_data
from gehm.losses.loss_functions import *
from gehm.datasets.nx_datasets import nx_dataset_onelevel
import pytest
import numpy as np
import torch


@pytest.mark.losses
def test_first_deg_loss(create_test_data):


    sim=np.zeros([3,3])
    sim[0,1]=1
    sim[0,2]=1
    sim[1,0]=1
    sim[2,0]=1

    sim2=np.zeros([3,3])
    sim2[0,1]=1
    sim2[0,2]=1
    sim2[1,0]=1
    sim2[2,0]=1

    pos=np.zeros([3,2])
    pos[1,:]=[-1,0]
    pos[2,:]=[1,0]

    pos2=np.zeros([3,2])
    pos2[1,:]=[3,0]
    pos2[2,:]=[1,0]

    sim=torch.as_tensor(sim)
    pos=torch.as_tensor(pos)
    pos2=torch.as_tensor(pos2)

    first_deg_loss = FirstDegLoss()

    assert first_deg_loss(pos, sim) < first_deg_loss(pos2, sim), "Loss of pos1 should be less than pos2, pos1-pos2 is {}".format(
        first_deg_loss(pos, sim)-first_deg_loss(pos2, sim)
    )

    second_deg_loss = SecondDegLoss()


    G, G_undir = create_test_data

    dataset = nx_dataset_onelevel(G)
    nodes, sim1, sim2 = dataset[0:6]

    positions = torch.as_tensor(np.random.rand(sim1.shape[0], 2))

    # Test size exception
    with pytest.raises(ValueError):
        first_deg_loss(positions[0 : positions.shape[0] - 2, :], sim1)


@pytest.mark.losses
def test_second_deg_loss(create_test_data):

    G, G_undir = create_test_data

    dataset = nx_dataset_onelevel(G)
    second_deg_loss = SecondDegLoss()
    nodes, sim1, sim2 = dataset[0:6]

    positions = torch.as_tensor(np.random.rand(sim1.shape[0], 2))

    # Test size exception
    with pytest.raises(ValueError):
        second_deg_loss(positions[0 : positions.shape[0] - 2, :], sim1)

    positions = torch.as_tensor(np.ones([sim1.shape[0], 2]))
    assert second_deg_loss(positions, sim1) < 0, "Neg loss should be <0, is {}".format(
        second_deg_loss(sim1, positions)
    )

