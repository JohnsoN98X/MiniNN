import numpy as np
from .base import Layer

class Tanh(Layer):
    '''
    Hyperbolic tangent (TanH) layer. 
    '''
    def forward(self, x):
        '''
        performs forward propagation. calculated as np.tanh(x).
        '''

        self.tanh = np.tanh(x)
        return self.tanh

    def backward(self, grad_outputs):
        '''
        perdorms backward propagation.
        '''
        grad_inputs = grad_outputs * (1 - self.tanh**2)
        return grad_inputs
    