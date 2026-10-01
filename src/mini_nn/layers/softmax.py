from .base import Layer
import numpy as np


class Softmax(Layer):
    '''
    Softmax layer.
    '''

    def forward(self, x):
        '''
        Performs forward propagation.
        '''
        shifted_x = x - np.max(x, axis=-1, keepdims=True)

        exponents = np.exp(shifted_x)
        exponent_sum = np.sum(exponents, axis=-1, keepdims=True)

        self.softmax = exponents / exponent_sum
        return self.softmax

    def backward(self, grad_outputs):
        '''
        Performs backward propagation.
        '''
        dot = np.sum(
            grad_outputs * self.softmax,
            axis=-1,
            keepdims=True
        )

        grad_inputs = self.softmax * (grad_outputs - dot)
        return grad_inputs
