import numpy as np


# koristi Least Squares Loss
class LSLoss:
    def calculate_loss(self, y_hat, Y):
        return np.mean(np.square(Y - y_hat))

    def calculate_loss_derivative(self, y_hat, Y):
        return 2 * (y_hat - Y) / Y.shape[0]

