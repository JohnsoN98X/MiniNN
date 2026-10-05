from .layers.base import Layer

class Sequential(Layer):
    '''
    Container for a sequence of neural network layers.
    Layers are applied in order during forward propagation and in reverse order during backpropagation.

    Parameters
    ----------
    layers: list
        Ordered list of layers.
    '''
    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, x):
        '''
        Passes input through all layers sequentially.
        '''
        for layer in self.layers:
            x = layer(x)

        return x

    def backward(self, grad_outputs):
        '''
        Propagates gradients backward through all layers. 
        '''
        grad = grad_outputs
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad