import numpy as np
from mini_nn.losses import BCEWithLogitLoss

# --- FORWARD PROPAGATION TEST ---
def test_bcewl_forward():
    loss_func = BCEWithLogitLoss()

    y_true = np.array([1.0, 0.0])
    y_pred = np.array([0.0, 0.0])

    loss = loss_func(y_true, y_pred)
    expected_loss = -np.log(0.5)

    assert np.allclose(loss, expected_loss)


# --- BACKPROPAGATION TEST ---
def test_bcewl_backward():
    loss_func = BCEWithLogitLoss()

    y_true = np.array([1.0, 0.0])
    y_pred = np.array([0.0, 0.0])

    expected_grad = np.array([-0.25, 0.25])

    # --- forward first ---
    loss_func(y_true, y_pred)

    # --- backward ---
    grad = loss_func.backward()

    # --- assert ---
    assert np.allclose(grad, expected_grad)




