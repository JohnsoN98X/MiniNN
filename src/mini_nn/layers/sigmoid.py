import numpy as np
from .base import Layer

class Sigmoid(Layer):
    '''
    Sigmoid layer. calculates 1/(1+e^(-x))
    '''

    def forward(self, x):
        """
        Performs forward propagation.
        """
        self.sigmoid = 1 / (1 + np.exp(-x))
        return self.sigmoid
    

    def backward(self, grad_outputs):
        '''
        Perform backward propagation.
        '''
        grad_inputs = grad_outputs * (self.sigmoid * (1-self.sigmoid))
        return grad_inputs