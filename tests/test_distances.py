from gehm.utils.distances import *
from tests.test_data import create_test_data

import pytest
import numpy as np
import torch
from numpy import cos, sin

@pytest.mark.distances
def test_nx_second_order_proximity(create_test_data):

    G,G_undir=create_test_data
    nodes=np.array(G.nodes)


    # Test 1: Ordering when subset proximity
    sel1=nodes[[0,1,2,3]]
    sel2=nodes[[0,3,2,1]]
    prox1=nx_second_order_proximity(G,sel1,False)
    prox2=nx_second_order_proximity(G,sel2,False)
    prox1=np.round(prox1,5)
    prox2=np.round(prox2,5)
    assert (prox1[0,1]==prox2[0,3]), "Ordering problem, {} != {}".format(prox1[0,1],prox2[0,3])
    assert (prox1[1,2]==prox2[3,2]), "Ordering problem, {} != {}".format(prox1[0,1],prox2[0,3])


    # Test 2: Ordering when whole network proximity
    prox1=nx_second_order_proximity(G,sel1,True)
    prox2=nx_second_order_proximity(G,sel2,True)
    prox1=np.round(prox1,5)
    prox2=np.round(prox2,5)
    assert (prox1[1,3]==prox2[3,3]), "Ordering problem, {} != {}".format(prox1[1,3],prox2[3,3])
    assert (prox1[3,1]==prox2[1,1]), "Ordering problem, {} != {}".format(prox1[3,1],prox2[1,1])

    # Test 3+4: Without row normalization
    prox1=nx_second_order_proximity(G,sel1,False, norm_rows=False)
    prox2=nx_second_order_proximity(G,sel2,False, norm_rows=False)
    prox1=np.round(prox1,5)
    prox2=np.round(prox2,5)
    assert (prox1[0,1]==prox2[0,3]), "Ordering problem, {} != {}".format(prox1[0,1],prox2[0,3])
    assert (prox1[1,2]==prox2[3,2]), "Ordering problem, {} != {}".format(prox1[0,1],prox2[0,3])

    prox1=nx_second_order_proximity(G,sel1,True, norm_rows=False)
    prox2=nx_second_order_proximity(G,sel2,True, norm_rows=False)
    prox1=np.round(prox1,5)
    prox2=np.round(prox2,5)
    assert (prox1[1,3]==prox2[3,3]), "Ordering problem, {} != {}".format(prox1[0,1],prox2[0,3])
    assert (prox1[3,1]==prox2[1,1]), "Ordering problem, {} != {}".format(prox1[0,1],prox2[0,3])

    # Test 5+6: Without row normalization, but with batch normalization
    prox1=nx_second_order_proximity(G,sel1,False, norm_rows=False, norm_rows_in_sample=True)
    prox2=nx_second_order_proximity(G,sel2,False, norm_rows=False, norm_rows_in_sample=True)
    prox1=np.round(prox1,5)
    prox2=np.round(prox2,5)
    assert (prox1[0,1]==prox2[0,3]), "Ordering problem, {} != {}".format(prox1[0,1],prox2[0,3])
    assert (prox1[1,2]==prox2[3,2]), "Ordering problem, {} != {}".format(prox1[0,1],prox2[0,3])

    prox1=nx_second_order_proximity(G,sel1,True, norm_rows=False, norm_rows_in_sample=True)
    prox2=nx_second_order_proximity(G,sel2,True, norm_rows=False, norm_rows_in_sample=True)
    prox1=np.round(prox1,5)
    prox2=np.round(prox2,5)
    assert (prox1[1,3]==prox2[3,3]), "Ordering problem, {} != {}".format(prox1[0,1],prox2[0,3])
    assert (prox1[3,1]==prox2[1,1]), "Ordering problem, {} != {}".format(prox1[0,1],prox2[0,3])


    # Test 7: Whole network, but return batch order
    prox1=nx_second_order_proximity(G,sel1,True, norm_rows=True, to_batch=True)
    prox2=nx_second_order_proximity(G,sel2,True, norm_rows=True, to_batch=True)
    prox1=np.round(prox1,5)
    prox2=np.round(prox2,5)
    assert (prox1[1,0]==prox2[3,0]), "Ordering problem, {} != {}".format(prox1[1,0],prox2[3,0])
    assert (prox1[3,1]==prox2[1,3]), "Ordering problem, {} != {}".format(prox1[3,1],prox2[1,3])

    # Test 8: Whole network, but return batch order, no norm
    prox1=nx_second_order_proximity(G,sel1,True, norm_rows=False, to_batch=True)
    prox2=nx_second_order_proximity(G,sel2,True, norm_rows=False, to_batch=True)
    prox1=np.round(prox1,5)
    prox2=np.round(prox2,5)
    assert (prox1[1,0]==prox2[3,0]), "Ordering problem, {} != {}".format(prox1[1,0],prox2[3,0])
    assert (prox1[3,1]==prox2[1,3]), "Ordering problem, {} != {}".format(prox1[3,1],prox2[1,3])



    # Now repeat everything with an undirected graph:

    G=G_undir
    nodes=np.array(G.nodes)


    # Test 1: Ordering when subset proximity
    sel1=nodes[[0,1,2,3]]
    sel2=nodes[[0,3,2,1]]
    prox1=nx_second_order_proximity(G,sel1,False)
    prox2=nx_second_order_proximity(G,sel2,False)
    prox1=np.round(prox1,5)
    prox2=np.round(prox2,5)
    assert (prox1[0,1]==prox2[0,3]), "Ordering problem, {} != {}".format(prox1[0,1],prox2[0,3])
    assert (prox1[1,2]==prox2[3,2]), "Ordering problem, {} != {}".format(prox1[0,1],prox2[0,3])


    # Test 2: Ordering when whole network proximity
    prox1=nx_second_order_proximity(G,sel1,True)
    prox2=nx_second_order_proximity(G,sel2,True)
    prox1=np.round(prox1,5)
    prox2=np.round(prox2,5)
    assert (prox1[1,3]==prox2[3,3]), "Ordering problem, {} != {}".format(prox1[0,1],prox2[0,3])
    assert (prox1[3,1]==prox2[1,1]), "Ordering problem, {} != {}".format(prox1[0,1],prox2[0,3])

    # Test 3+4: Without row normalization
    prox1=nx_second_order_proximity(G,sel1,False, norm_rows=False)
    prox2=nx_second_order_proximity(G,sel2,False, norm_rows=False)
    prox1=np.round(prox1,5)
    prox2=np.round(prox2,5)
    assert (prox1[0,1]==prox2[0,3]), "Ordering problem, {} != {}".format(prox1[0,1],prox2[0,3])
    assert (prox1[1,2]==prox2[3,2]), "Ordering problem, {} != {}".format(prox1[0,1],prox2[0,3])

    prox1=nx_second_order_proximity(G,sel1,True, norm_rows=False)
    prox2=nx_second_order_proximity(G,sel2,True, norm_rows=False)
    prox1=np.round(prox1,5)
    prox2=np.round(prox2,5)
    assert (prox1[1,3]==prox2[3,3]), "Ordering problem, {} != {}".format(prox1[0,1],prox2[0,3])
    assert (prox1[3,1]==prox2[1,1]), "Ordering problem, {} != {}".format(prox1[0,1],prox2[0,3])

    # Test 5+6: Without row normalization, but with batch normalization
    prox1=nx_second_order_proximity(G,sel1,False, norm_rows=False, norm_rows_in_sample=True)
    prox2=nx_second_order_proximity(G,sel2,False, norm_rows=False, norm_rows_in_sample=True)
    prox1=np.round(prox1,5)
    prox2=np.round(prox2,5)
    assert (prox1[0,1]==prox2[0,3]), "Ordering problem, {} != {}".format(prox1[0,1],prox2[0,3])
    assert (prox1[1,2]==prox2[3,2]), "Ordering problem, {} != {}".format(prox1[0,1],prox2[0,3])

    prox1=nx_second_order_proximity(G,sel1,True, norm_rows=False, norm_rows_in_sample=True)
    prox2=nx_second_order_proximity(G,sel2,True, norm_rows=False, norm_rows_in_sample=True)
    prox1=np.round(prox1,5)
    prox2=np.round(prox2,5)
    assert (prox1[1,3]==prox2[3,3]), "Ordering problem, {} != {}".format(prox1[0,1],prox2[0,3])
    assert (prox1[3,1]==prox2[1,1]), "Ordering problem, {} != {}".format(prox1[0,1],prox2[0,3])


    # Test 7: Whole network, but return batch order
    prox1=nx_second_order_proximity(G,sel1,True, norm_rows=True, to_batch=True)
    prox2=nx_second_order_proximity(G,sel2,True, norm_rows=True, to_batch=True)
    prox1=np.round(prox1,5)
    prox2=np.round(prox2,5)
    assert (prox1[1,0]==prox2[3,0]), "Ordering problem, {} != {}".format(prox1[1,0],prox2[3,0])
    assert (prox1[3,1]==prox2[1,3]), "Ordering problem, {} != {}".format(prox1[3,1],prox2[1,3])

    # Test 8: Whole network, but return batch order, no norm
    prox1=nx_second_order_proximity(G,sel1,True, norm_rows=False, to_batch=True)
    prox2=nx_second_order_proximity(G,sel2,True, norm_rows=False, to_batch=True)
    prox1=np.round(prox1,5)
    prox2=np.round(prox2,5)
    assert (prox1[1,0]==prox2[3,0]), "Ordering problem, {} != {}".format(prox1[1,0],prox2[3,0])
    assert (prox1[3,1]==prox2[1,3]), "Ordering problem, {} != {}".format(prox1[3,1],prox2[1,3])



@pytest.mark.distances
def test_embedding_first_order_proximity():

    positions=torch.as_tensor(np.random.rand(4,2))

    pos1=positions[1,:]
    pos2=positions[3,:]

    distance=np.round(1-np.sqrt(np.square(abs(pos1-pos2)).sum()),4)
    sim_matrix=np.round(embedding_first_order_proximity(positions,False),4)
    assert (distance==sim_matrix[1,3]), "Distances do not fit, {} != {}".format(distance,sim_matrix[1,3])

