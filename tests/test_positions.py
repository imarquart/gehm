from tests.test_data import create_test_data
from gehm.model.positions import *
import pytest
import numpy as np
import torch
from numpy import cos, sin


@pytest.mark.position_embeddings
def test_circle():

    expected_output = np.zeros([3, 2])
    expected_output[0, :] = [cos(0), sin(0)]
    expected_output[1, :] = [cos(180), sin(180)]
    expected_output[2, :] = [cos(90), sin(90)]

    # Test 1: Scaling between 0 and 1
    c1 = Circle(max_value=1)
    test_input = torch.as_tensor(np.zeros(3))
    test_input[0] = 0
    test_input[1] = 0.5
    test_input[2] = 0.25
    output = c1(test_input.unsqueeze(1))
    assert (
        output == torch.tensor(expected_output)
    ).all(), "output {} not equal {}".format(output, expected_output)
    # Test 2: Scaling between 0 and 360
    c2 = Circle(max_value=360)
    test_input = torch.as_tensor(np.zeros(3))
    test_input[0] = 0
    test_input[1] = 180
    test_input[2] = 90
    output = c2(test_input.unsqueeze(1))
    assert (
        output == torch.tensor(expected_output)
    ).all(), "output {} not equal {}".format(output, expected_output)


@pytest.mark.position_embeddings
def test_circle_batch(create_test_data):

    expected_output = np.zeros([3, 2])
    expected_output[0, :] = [cos(0), sin(0)]
    expected_output[1, :] = [cos(180), sin(180)]
    expected_output[2, :] = [cos(90), sin(90)]
    expected_output = torch.tensor([expected_output, expected_output, expected_output])

    # Test 1: Scaling between 0 and 1
    c1 = Circle(max_value=1)
    test_input = np.zeros([3, 1])
    test_input[0, :] = 0
    test_input[1, :] = 0.5
    test_input[2, :] = 0.25
    test_input = torch.tensor([test_input, test_input, test_input])
    output = c1(test_input)
    assert (output == expected_output).all(), "output {} not equal {}".format(
        output, expected_output
    )

    # Test 2: Scaling between 0 and 360
    c2 = Circle(max_value=360)
    test_input = np.zeros([3, 1])
    test_input[0, :] = 0
    test_input[1, :] = 180
    test_input[2, :] = 90
    test_input = torch.tensor([test_input, test_input, test_input])
    output = c2(test_input)
    assert (output == expected_output).all(), "output {} not equal {}".format(
        output, expected_output
    )


@pytest.mark.position_embeddings
def test_disk(create_test_data):

    # Test 1: Scaling between 0 and 1 in numpy
    expected_output = np.zeros([3, 2])
    expected_output[0, :] = [1, 0]
    expected_output[1, :] = [1, 0]
    expected_output[2, :] = [0.125, 0.1]
    expected_output = torch.tensor(expected_output)

    d1 = Disk(max_value=1)
    test_input = np.zeros([3, 2])
    test_input[0, :] = [1, 0]
    test_input[1, :] = [100, 0]
    test_input[2, :] = [0.125, 0.1]
    test_input = torch.as_tensor(test_input)
    output = d1(test_input)
    assert (output == expected_output).all(), "output {} not equal {}".format(
        output, expected_output
    )

    # Test 2: Scaling between 0 and 50 in numpy
    expected_output = np.zeros([3, 2])
    expected_output[0, :] = [1, 0]
    expected_output[1, :] = [50, 0]
    expected_output[2, :] = [0.125, 0.1]
    expected_output = torch.tensor(expected_output)
    d1 = Disk(max_value=50)
    test_input = np.zeros([3, 2])
    test_input[0, :] = [1, 0]
    test_input[1, :] = [100, 0]
    test_input[2, :] = [0.125, 0.1]
    test_input = torch.as_tensor(test_input)
    output = d1(test_input)
    assert (output == expected_output).all(), "output {} not equal {}".format(
        output, expected_output
    )

