import numpy as np
from .base import Loss

class BCELoss(Loss):
    '''
    Binary cross enthropy loss. 
    '''
    def forward(self, y_true, y_pred):

        eps = 1e-10
        self.y_pred = np.clip(self.y_pred, eps, 1-eps)

        bce = -np.mean(self.y_true * np.log(self.y_pred)
                        + (1-self.y_true) * np.log(1-self.y_pred))
        return bce

    def backward(self):
        grad_pred = (1/self.n) * ((self.y_pred - self.y_true)/
                             (self.y_pred * (1-self.y_pred)))
        return grad_pred