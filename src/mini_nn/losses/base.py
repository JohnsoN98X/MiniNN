class Loss:
    def __call__(self, y_true, y_pred):
        if y_true.shape != y_pred.shape:
            raise ValueError('y_true and y_pred must be the same shape')

        self.y_true = y_true
        self.y_pred = y_pred
        self.n = y_true.size
        
        return self.forward(y_true, y_pred)

    def forward(self, y_true, y_pred):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError