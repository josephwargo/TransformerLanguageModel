import numpy as np

class layer_norm(object):
    def __init__(self, input_size):
        self.gamma = np.ones(shape=input_size)
        self.beta = np.zeros(shape=input_size)

        self.prev_layer_hidden_state = None
        self.x_std_dev = None
        self.x_hat_val = None

    def x_hat(self, x, train=False):
        if train:
            self.prev_layer_hidden_state = x

        x_mean = np.mean(x, axis=-1, keepdims=True)
        x_std_dev = np.sqrt(np.var(x, axis=-1, keepdims=True) + 1e-5)

        if train:
            self.x_std_dev = x_std_dev

        x_hat = (x - x_mean) / x_std_dev

        return x_hat
    
    def forward_pass(self, x, train=False):
        
        self.x_hat_val = self.x_hat(x, train)
        
        layer_normed = self.gamma * self.x_hat_val + self.beta
        
        return layer_normed
    
    def backward_pass(self, dL_dY):
        
        # gradients to update gamma and beta
        dL_dgamma = (dL_dY.transpose(0,2,1) @ self.x_hat_val).sum(axis=(0,1))
        dL_dgamma = (dL_dY * self.x_hat_val).sum(axis=(0,1))
        dL_dbeta = dL_dY.sum(axis=(0,1))

        # gradient to pass back into normalization step
        dL_dY = dL_dY * self.gamma

        # passing gradient back through normalization
        # TODO: write this
        # dL_dY = dL_dY / self.x_std_dev