import numpy as np

class layer_norm(object):
    def __init__(self, input_size):
        self.gamma = np.ones(shape=input_size)
        self.beta = np.zeros(shape=input_size)

    def x_hat(self, x):
        x_mean = np.mean(x)
        x_std_dev = np.std(x)

        x_hat = (x - x_mean) / np.sqrt(x_std_dev + 1e-5)

        return x_hat
    
    def layer_norm(self, x):
        return self.gamma * self.x_hat(x) + self.beta
