import numpy as np

class SoftmaxLayer(object):
    def __init__(self, name):
        super(SoftmaxLayer, self).__init__()
        self.name = name

    def has_params(self):
        return False

    def forward(self, X):
        """
        Forward message.
        :param X: inputs (outputs of the previous layer), shape (n_samples, N)
        :return: layer output, shape (n_samples, N)
        """
        X = np.exp(X - np.max(X, axis=1, keepdims=True))
        return X/ np.sum(X, axis=1, keepdims=True)

    def delta(self, Y, delta_next):
        """
        Computes delta (dl/d(layer inputs)), based on delta from the following layer. The computations involve backward
        message.
        :param Y: output of this layer (i.e., input of the next), shape (n_samples, N)
        :param delta_next: delta vector backpropagated from the following layer, shape (n_samples, N)
        :return: delta vector from this layer, shape (n_samples, N)
        """

        dot = np.sum(delta_next * Y, axis=1, keepdims=True)
        return Y * (delta_next - dot)
        
