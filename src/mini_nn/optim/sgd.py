import numpy as np

class SGD:
    def __init__(self, layers, learning_rate: float):
        '''
        Performs Stochastic Gradient Descent.

        Parameters
        ----------
        layers: 
            ... 
        
        learning_rate: int
            learning rate of the optimizer.
        '''
        self.layers = layers
        self.lr = learning_rate


    def step(self):
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                layer.weight -= self.lr * layer.dw
                layer.bias -= self.lr * layer.db


