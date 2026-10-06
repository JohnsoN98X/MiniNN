import numpy as np
from mini_nn.layers import ReLu

def test_relu_forward():
    relu = ReLu()
    x = np.array([-2, 0, 3])
    expected = np.array([0, 0, 3])

    output = relu(x)
    assert np.allclose(output, expected)

def test_relu_backward():
    relu = ReLu()

    x = np.array([-2, 0, 3])
    grad_outputs = np.array([1, 1, 1])

    relu(x)
    grad = relu.backward(grad_outputs)
    expected = np.array([0,0,0])

    assert np.allclose(grad, expected)

