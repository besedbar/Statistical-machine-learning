import numpy as np

class LossCrossEntropy(object):
    def __init__(self, name):
        super(LossCrossEntropy, self).__init__()
        self.name = name

    def forward(self, X, T):
        """
        Forward message.
        :param X: loss inputs (outputs of the previous layer), shape (n_samples, n_inputs), n_inputs is the same as
        the number of classes
        :param T: one-hot encoded targets, shape (n_samples, n_inputs)
        :return: layer output, shape (n_samples, 1)
        """
        eps = 1e-15
        X_clipped = np.clip(X, eps, 1 - eps)
        losses = -np.sum(T * np.log(X_clipped), axis=1, keepdims=True)
        return losses
       

    def delta(self, X, T):
        """
        Computes delta vector for the output layer.
        :param X: loss inputs (outputs of the previous layer), shape (n_samples, n_inputs), n_inputs is the same as
        the number of classes
        :param T: one-hot encoded targets, shape (n_samples, n_inputs)
        :return: delta vector from the loss layer, shape (n_samples, n_inputs)
        """
        eps = 1e-15
        X_clipped = np.clip(X, eps, 1 - eps)
        # dL/dX = -T / X (po prvcích)
        return -(T / X_clipped)
        


class LossCrossEntropyForSoftmaxLogits(object):
    def __init__(self, name):
        super(LossCrossEntropyForSoftmaxLogits, self).__init__()
        self.name = name

    def forward(self, X, T):
        """
        Forward message.
        :param X: loss inputs (outputs of the previous layer), shape (n_samples, n_inputs), n_inputs is the same as
        the number of classes
        :param T: one-hot encoded targets, shape (n_samples, n_inputs)
        :return: layer output, shape (n_samples, 1)
        """
        #return -np.sum(T * X, axis=1, keepdims=True).mean() + np.log(np.sum(np.exp(X - np.max(X, axis=1, keepdims=True)), axis=1, keepdims=True)).mean() + np.max(X, axis=1, keepdims=True).mean()
        max_x = np.max(X, axis=1, keepdims=True)
        X_shifted = X - max_x

        # log-sum-exp
        log_sum_exp = np.log(np.sum(np.exp(X_shifted), axis=1, keepdims=True))  # (N, 1)

        # logit správné třídy (taky shifted)
        correct_logits = np.sum(T * X_shifted, axis=1, keepdims=True)          # (N, 1)

        losses = log_sum_exp - correct_logits
        return losses

    def delta(self, X, T):
        """
        Computes delta vector for the output layer.
        :param X: loss inputs (outputs of the previous layer), shape (n_samples, n_inputs), n_inputs is the same as
        the number of classes
        :param T: one-hot encoded targets, shape (n_samples, n_inputs)
        :return: delta vector from the loss layer, shape (n_samples, n_inputs)
        """
        
        max_x = np.max(X, axis=1, keepdims=True)
        X_shifted = X - max_x
        exp_x = np.exp(X_shifted)
        softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return softmax - T