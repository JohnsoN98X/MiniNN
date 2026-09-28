import numpy as np
from .base import Loss

class MSELoss(Loss):
    def forward(self, y_true: np.array, y_pred: np.array):
        if not y_true.shape == y_pred.shape:
            raise ValueError('y_true and y_pred must be the same shape')
        
        self.true = y_true
        self.pred = y_pred

        mse = np.mean((self.pred - self.true)**2)
        return mse

    def backward(self):
        n = self.pred.size
        grad_pred = (2/n) * (self.pred - self.true)
        return grad_pred
    
        