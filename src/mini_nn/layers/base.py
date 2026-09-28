class Layer:
    """
    Base class for all neural network layers.

    Defines the common interface that every layer should implement:
    forward propagation and backward propagation.
    """

    def __call__(self, X):
        """
        Apply the layer to the input.

        Parameters
        ----------
        X : np.ndarray
            Input to the layer.

        Returns
        -------
        np.ndarray
            Output of the layer.
        """
        return self.forward(X)

    def forward(self, X):
        """
        Perform forward propagation.

        Parameters
        ----------
        X : np.ndarray
            Input to the layer.

        Returns
        -------
        np.ndarray
            Output of the layer.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError

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

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError