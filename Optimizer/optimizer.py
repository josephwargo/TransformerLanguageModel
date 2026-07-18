import numpy as np
import cupy as cp
import scheduler as sch

class optimizer(object):
####################################
# Initializations #
####################################
    def __init__(self, weight_decay=.01, adamW=False, input_shape=None, output_shape=None):
        self.weight_decay = weight_decay
        # arrays to store 
        
        if adamW:
            # constants
            self.beta1 = .9
            self.beta2 = .999
            self.epsilon = 1e-8
            self.t = 1
            # momentum
            self.md_layer_weights = cp.zeros(shape=(input_shape,output_shape)).astype(cp.float32)
            self.vd_layer_weights = cp.zeros(shape=(input_shape,output_shape)).astype(cp.float32)
            # variance
            self.md_bias = cp.zeros(shape=(output_shape)).astype(cp.float32)         
            self.vd_bias = cp.zeros(shape=(output_shape)).astype(cp.float32)

    def update_adam(self, learning_rate, dL_dW, dL_db):
            print("adam")
            # doing ^t on beta1 and beta2 once per step
            b1T = self.beta1**self.t
            b2T = self.beta2**self.t
            
            # layer weights updates
            # momentum stored
            self.md_layer_weights *= self.beta1
            self.md_layer_weights += (1-self.beta1)*dL_dW
            
            # variance stored
            self.vd_layer_weights *= self.beta2
            self.vd_layer_weights += (1-self.beta2)*(dL_dW**2)
            
            md_update_val = self.md_layer_weights/(1-b1T)
            vd_update_val = self.vd_layer_weights/(1-b2T)

            update_val = schedule_multiplier*(((learning_rate*md_update_val) / (cp.sqrt(vd_update_val) + self.epsilon)) + (self.weight_decay*self.layer_weights))

            # updates (incl. corrections and Adam)
            self.layer_weights += -learning_rate * (self.md_layer_weights / (1-b1T) / (cp.sqrt(self.vd_layer_weights / (1-b2T))+self.epsilon))

            # bias updates
            # momentum stored
            self.md_bias *= self.beta1
            self.md_bias += (1-self.beta1)*self.bias_updates
            # RMSProp stored
            self.vd_bias *= self.beta2
            self.vd_bias += (1-self.beta2)*(self.bias_updates**2)
            # update (incl. corrections and Adam)
            self.bias += -self.learning_rate * (self.md_bias / (1-b1T) / (cp.sqrt(self.vd_bias / (1-b2T))+self.epsilon))

            self.t+=1 # increment