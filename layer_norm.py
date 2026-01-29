import numpy as np

class layer_norm(object):
    def __init__(self, input_size):
        self.gamma = None # needs to be the size of the input shape
        self.beta = None # needs to be the size of the input shape

    def x_hat(x):
        x_mean = np.mean(x)
        x_std_dev = np.std(x)

        x_hat = (x - x_mean) / np.sqrt(x_std_dev + 1e-5)

        return x_hat
    
    def affine_trans(x_hat):
        pass