import numpy as np
class neuron_layer(object):
    def __init__(self, input_shape, output_shape, activation, batch_size,
                 clip_val, learning_rate, adam=False):

        # layer info 
        self.input_shape = input_shape # length of vector that will be inputted into the weights
        self.output_shape = output_shape # desired length of weight output (and therefore layer output)
        layer_xavier = np.sqrt(2/(self.input_shape+self.output_shape)) # value used to determine optimal
        self.layer_weights = np.random.normal(0,layer_xavier, size=(self.input_shape,self.output_shape)).astype(np.float32) # optimal weight initialization via xavier

        self.bias = np.zeros(shape=(output_shape)).astype(np.float32) # initializing bias as 0s - no xavier here

        # activation
        self.activation = activation # activation function that will be applied to the output of the weights

        # hyperparameters
        self.batch_size = batch_size # number of instances per training run
        self.clip_val = clip_val # used to set the upper and lower bounds for what we will let weight updates hit
        self.learning_rate = learning_rate # value we multiply by the weight updates given to us via gradient descent

        # adam
        self.adam = adam
        if adam:
            # constants
            self.beta1 = .9
            self.beta2 = .999
            self.epsilon = 1e-8
            self.t = 1
            # arrays to store 
            self.md_layer_weights = np.zeros(shape=(self.input_shape,self.output_shape)).astype(np.float32)
            self.vd_layer_weights = np.zeros(shape=(self.input_shape,self.output_shape)).astype(np.float32)

            self.md_bias = np.zeros(shape=(output_shape)).astype(np.float32)         
            self.vd_bias = np.zeros(shape=(output_shape)).astype(np.float32)
    
    def update(self, num_steps):

        # normalizing
        self.layer_weight_updates /= num_steps
        self.bias_updates /= num_steps

        # clipping
        np.clip(self.layer_weight_updates, -self.clip_val, self.clip_val, out=self.layer_weight_updates)
        np.clip(self.bias_updates, -self.clip_val, self.clip_val, out=self.bias_updates)

        # adam
        if self.adam:
            self.update_adam()
        
        else:
            self.layer_weights += -self.learning_rate*self.layer_weight_updates
            self.bias += -self.learning_rate*self.bias_updates


    def update_adam(self):
        
        # doing ^t on beta1 and beta2 once per step
        b1T = self.beta1**self.t
        b2T = self.beta2**self.t
        
        # layer weights
        # momentum stored
        self.md_layer_weights *= self.beta1
        self.md_layer_weights += (1-self.beta1)*self.layer_weight_updates
        # RMSProp stored
        self.vd_layer_weights *= self.beta2
        self.vd_layer_weights += (1-self.beta2)*(self.layer_weight_updates**2)
        # updates (incl. corrections and Adam)
        self.layer_weights += -self.learning_rate * (self.md_layer_weights / (1-b1T) / (np.sqrt(self.vd_layer_weights / (1-b2T))+self.epsilon))

        # bias
        # momentum stored
        self.md_bias *= self.beta1
        self.md_bias += (1-self.beta1)*self.bias_updates
        # RMSProp stored
        self.vd_bias *= self.beta2
        self.vd_bias += (1-self.beta2)*(self.bias_updates**2)
        # update (incl. corrections and Adam)
        self.bias += -self.learning_rate * (self.md_bias / (1-b1T) / (np.sqrt(self.vd_bias / (1-b2T))+self.epsilon))

        self.t+=1 # increment