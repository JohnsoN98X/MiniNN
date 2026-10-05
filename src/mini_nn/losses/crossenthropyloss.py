import numpy as np
from .base import Loss

class CrossEnthropyLoss(Loss):
    '''
    Cross Enthropy Loss.

    Parameters
    ----------
    y_true: np.ndarray 
        mXn one-hot matri of the real values.
    y_pred: np.ndarray
        mXn matrix of the predicted probabilities
    '''
    def forward(self, y_true, y_pred):
        '''
        Performs forward propagation.

        Returns
        ----------
        ce: int
            cross-enthropy loss value 
        '''
        self.n = y_true.shape[0]

        ce = -np.mean(
            np.sum(y_true * np.log(y_pred), axis=1)
        )
        return ce

    def backward(self):
        '''
        Performs back propagation.

        Returns
        ----------
        grad_pred:
            loss gradient with respect to the loss input
        '''
        grad_pred = -(self.y_true / self.y_pred) / self.n
        return grad_pred