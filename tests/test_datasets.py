from tests.test_data import create_test_data
from gehm.datasets.nx_datasets import *
import pytest
import numpy as np
import torch


@pytest.mark.datasets
def test_nx_dataset_onelevel(create_test_data):

    G,G_undir=create_test_data

    # Initializations
    focal_node = list(G.nodes)[0]
    adj=[x['weight'] for x in dict(G[focal_node]).values()]

    dataset=nx_dataset_onelevel(G,norm_rows=False, proximities=1)
    node, sim1, sim2 = dataset[0]
    assert sim1.sum()==np.sum(adj)

    dataset=nx_dataset_onelevel(G,norm_rows=False, proximities=[1,2])
    node, sim1, sim2 = dataset[0]
    assert sim1.sum()==np.sum(adj)

    dataset=nx_dataset_onelevel(G,norm_rows=True, proximities=[1,2])
    node, sim1, sim2 = dataset[0]
    assert sim1.sum()==1
    assert sim2.sum()==1

    dataset=nx_dataset_onelevel(G,norm_rows=True, proximities=[2])
    node, sim1, sim2 = dataset[0]
    assert sim2.sum()==1