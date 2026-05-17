import numpy as np

class layer_norm(object):
####################################
# Initializations #
####################################
    def __init__(self, input_size):
        # self.batch_size = batch_size

        self.gamma = np.ones(shape=input_size)
        self.beta = np.zeros(shape=input_size)

        self.prev_layer_output = None
        self.x_std_dev = None
        self.x_hat_val = None

####################################
# Forward Pass #
####################################
    def x_hat(self, x, train=False):
        if train:
            self.prev_layer_output = x

        x_mean = np.mean(x, axis=-1, keepdims=True)
        x_std_dev = np.sqrt(np.var(x, axis=-1, keepdims=True) + 1e-5)

        if train:
            self.x_std_dev = x_std_dev

        x_hat = (x - x_mean) / x_std_dev

        return x_hat
    
    def forward_pass(self, x, train=False):
        
        x_hat_val = self.x_hat(x, train)
        
        if train:
            self.x_hat_val = x_hat_val

        layer_normed = self.gamma * x_hat_val + self.beta
        
        return layer_normed

####################################
# Backward Pass #
####################################    
    def backward_pass(self, learning_rate, dL_dY):
        # gradients to update gamma and beta
        dL_dgamma = (dL_dY * self.x_hat_val).sum(axis=(0,1))
        dL_dbeta = dL_dY.sum(axis=(0,1))

        # gradient to pass back into normalization step
        dL_dx_hat = dL_dY * self.gamma
 
        # passing gradient back through normalization
        dL_dx_hat_direct = dL_dx_hat * dL_dY.shape[-1]
        dL_dx_hat_mean_penalty = dL_dx_hat.sum(axis=-1, keepdims=True)
        
        dL_dx_hat_variance_penalty = (dL_dx_hat*self.x_hat_val).sum(axis=-1, keepdims=True) * self.x_hat_val
        
        dL_dx = (dL_dx_hat_direct - dL_dx_hat_mean_penalty - dL_dx_hat_variance_penalty) / (self.x_std_dev * dL_dY.shape[-1])

        self.update(learning_rate, dL_dgamma, dL_dbeta)

        return dL_dx

    def update(self, learning_rate, dL_dgamma, dL_dbeta):
        self.gamma += -learning_rate * dL_dgamma
        self.beta += -learning_rate * dL_dbeta