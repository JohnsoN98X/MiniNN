import numpy as np
from .base import Loss

class MSELoss(Loss):
    '''
    Mean Squared Error (L2 error)
    '''
    def forward(self, y_true: np.array, y_pred: np.array):
        mse = np.mean((self.y_pred - self.y_true)**2)
        return mse

    def backward(self):
        n = self.y_pred.size
        grad_pred = (2/n) * (self.y_pred - self.y_true)
        return grad_pred
    
        