import numpy as np
from .base import Layer

import numpy as np
from .base import Layer


class ReLu(Layer):
    """
    ReLU layer. Calculates max(0, input).
    """

    def forward(self, x):
        """
        Perform forward propagation.

        Parameters
        ----------
        x : np.ndarray
            Output of the previous layer.

        Returns
        -------
        np.ndarray
            ReLU output, calculated as np.maximum(0, x).
        """
        self.x = x
        relu = np.maximum(0, x)
        return relu

    def backward(self, grad_outputs):
        """
        Perform backward propagation.

        Parameters
        ----------
        grad_outputs : np.ndarray
            Gradient of the loss with respect to the layer's output.

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to the layer's input.
        """
        grad_inputs = grad_outputs * (self.x > 0)
        return grad_inputs
        