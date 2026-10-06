import numpy as np
from mini_nn.losses import MSELoss

# --- LOSS VALUE TEST ---
def test_mse():
    mse_func = MSELoss()

    # --- set pred and true vectors ---
    y_true = np.array([
        1,2,3,4,5,6,7
    ])
    y_pred = np.array([
        2,4,5,3,7,5,6
    ])

    # --- set expected output ---
    expected_loss = np.mean(
        np.power((y_pred - y_true), 2)
    )
    loss = mse_func(y_true, y_pred)

    # --- assert ---
    assert np.allclose(expected_loss, loss)


# --- LOSS BACKWARD TEST ---
def test_mse_backward():
    mse_func = MSELoss()

    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 4.0, 1.0])

    # --- forward propagation first ---
    loss = mse_func(y_true, y_pred)

    # --- gradients ---
    grad = mse_func.backward()
    expected_grad = (2 / y_true.size) * (y_pred - y_true)

    # --- assert ---
    assert np.allclose(grad, expected_grad)