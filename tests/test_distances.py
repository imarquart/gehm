from gehm.utils.distances import *
from tests.test_data import create_test_data

import pytest
import numpy as np
import torch

@pytest.mark.distances
def test_embedding_first_order_proximity():

    positions=torch.as_tensor(np.random.rand(4,2))

    pos1=positions[1,:]
    pos2=positions[3,:]

    distance=np.round(1-np.sqrt(np.square(abs(pos1-pos2)).sum()),4)
    sim_matrix=np.round(embedding_first_order_proximity(positions,False),4)
    assert (distance==sim_matrix[1,3]), "Distances do not fit, {} != {}".format(distance,sim_matrix[1,3])

