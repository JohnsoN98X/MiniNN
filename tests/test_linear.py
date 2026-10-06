import numpy as np
from mini_nn.layers import Linear

# --- FORWARD PROPAGATION TEST ---
def test_linear_forward():
    # --- initialize layer, wiehgts and bias ---
    layer = Linear(2,2)
    layer.weight = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    layer.bias = np.array([
        0.5, -0.5
    ])

    # --- set input (x), expected output, and real output ---
    x = np.array([1.0, 2.0])
    expected = np.array([
        7.5, 9.5
    ])
    output = layer(x)

    # --- assert ---
    assert np.allclose(expected, output)


# --- BACKPROPAGATION TEST ---
def test_linear_backward():
    # --- initialize layer, wiehgts and bias ---
    layer = Linear(2,2)
    layer.weight = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    layer.bias = np.array([
        -0.5, 0.5
    ])

    # --- set an input and porform forward propagation ---
    x = np.array([
        [1.0, 2.0]
    ])
    layer(x)

    # --- backpropagation ---
    grad_outputs = np.array([
        [1.0, 2.0]
    ])
    expected_dw = np.array([
        [1.0, 2.0],
        [2.0, 4.0]
    ])
    expected_db = np.array([
        1.0, 2.0
    ])
    expected_grad_inputs = np.array([
        [5.0, 11.0]
    ])
    grad_inputs = layer.backward(grad_outputs)

    assert np.allclose(layer.dw, expected_dw)
    assert np.allclose(layer.db, expected_db)
    assert np.allclose(grad_inputs, expected_grad_inputs)
