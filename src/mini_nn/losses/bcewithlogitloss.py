import numpy as np
from .base import Loss

class BCEWithLogitLoss(Loss):
    '''
    Binary cross enthropy with logit
    '''
    def forward(self, y_true, y_pred):

        self.sigmoid = 1/(1+np.exp(-y_pred))

        bcewl = np.mean(
            np.maximum(y_pred, 0) - y_pred*y_true
        + np.log(
            1 + np.exp(-np.abs(y_pred))
        )
        )
        return bcewl

    def backward(self):
        grad_pred = (1/self.n) * (self.sigmoid - self.y_true)
        return grad_pred