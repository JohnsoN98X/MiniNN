import numpy as np

class Linear:
    def __init__(self, in_features:int, out_features:int):
        """
        Initialize the linear layer.
        --- Parameters ---
        - in_features (int): the shape of the input features. 
        - out_features (int) - the shape of the output features.
        """
        self.in_features = in_features
        self.out_features = out_features

        self.weight = np.random.randn(in_features, out_features) * 0.01
        self.bias = np.zeros((1, out_features))


    def forward(self, X):
        """
        Performs forward propagation.

        Parameters
        ----------
        X (np.ndarray):
            the output of the previous layer.

        Returns
        ----------
        y (np.ndarray):
            X @ self.weight + self.bias
        """
        self.X = X
        y =  X @ self.weight + self.bias
        return y


    def backward(self, grad_outputs):
        """
        Takes the gradients of the following layer. 
        
        Parameters
        ----------
        grad_outputs : nd.array
            the loss gradients of the layer's output; grad_outputs = dL / dy

        Attributes
        ----------
        dw : np.ndarray
            The computed loss gradient with respect to the weights (dL/dW),
            calculated as self.X.T @ grad_outputs
        db : np.ndarray
            The computed loss gradient with respect to the bias (dL/db), 
            calculated as np.sum(grad_outputs, axis=0). 
        
        Returns
        -----------
        grad_inputs : np.ndarray
            Gradient of the loss with respect to the layer's input (dL/dX).
            This is passed to the previous layer as its grad_outputs.
        """
        self.dw = self.X.T @ grad_outputs
        self.db = np.sum(grad_outputs, axis=0, keepdims=True)

        grad_inputs = grad_outputs @ self.weight.T

        return grad_inputs
