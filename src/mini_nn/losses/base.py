class Loss:
    def __call__(self, y_true, y_pred):
        return self.forward(y_true, y_pred)

    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError