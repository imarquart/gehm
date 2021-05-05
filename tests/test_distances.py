from gehm.utils.position import *
import pytest
import numpy as np
import torch
from numpy import cos, sin

@pytest.mark.distances
def test_nx_second_order_proximity(create_test_data):

    expected_output=np.zeros([3,2])
    expected_output[0,:]=[cos(0),sin(0)]
    expected_output[1,:]=[cos(180),sin(180)]
    expected_output[2,:]=[cos(90),sin(90)]

    # Test 1: Scaling between 0 and 1 in numpy
    c1=circle(False, max_value=1)
    test_input=np.zeros(3)
    test_input[0]=0
    test_input[1]=0.5
    test_input[2]=0.25
    output=c1.embed(test_input)
    assert (output==torch.tensor(expected_output)).all(), "output {} not equal {}".format(output, expected_output)
    # Test 2: Scaling between 0 and 360 in numpy
    c2=circle(False, max_value=360)
    test_input=np.zeros(3)
    test_input[0]=0
    test_input[1]=180
    test_input[2]=90
    output=c2.embed(test_input)
    assert (output==torch.tensor(expected_output)).all(), "output {} not equal {}".format(output, expected_output)

    # Test 1: Scaling between 0 and 1 in pytorch
    test_input=torch.tensor(test_input)
    output=c2.embed(test_input)
    assert (output==torch.tensor(expected_output)).all(), "output {} not equal {}".format(output, expected_output)

