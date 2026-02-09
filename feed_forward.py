import numpy as np
import costs_and_activations as caa


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

        # hidden state
        self.hidden_state = None

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


####################################
# Forward Pass #
####################################
    def forward_pass(self, x, train=False):
        if self.activation==None:
            hidden_state = x @ self.layer_weights + self.bias
        else:
            hidden_state = caa.activation(self.activation, x @ self.layer_weights + self.bias)
        if train:
            self.hidden_state = hidden_state
        return hidden_state

####################################
# Backward Pass #
####################################

    def backward_pass(self, logits, Y, pad_token_ind=0):
        # if this is an output layer
        # TODO: clean up so output layer is True/False not activation as none. also valid for forward_pass
        if self.activation==None:
            # flattening and masking logits and Y
            Y_flat = Y.reshape(-1)
            mask = (Y_flat != pad_token_ind)
            Y_flat_masked = Y_flat[mask]
            logits_flat = logits.reshape(-1, logits.shape[-1])
            logits_flat_masked = logits_flat[mask]
            
            # grad for non-padded
            dL_dZ_active = caa.softmax_cross_entropy_grad(logits_flat_masked, Y_flat_masked)

            # reshaping to pre-flattened shape
            dL_dZ_flat = np.zeros_like(logits_flat)
            dL_dZ_flat[mask] = dL_dZ_active

            dL_dZ = dL_dZ_flat.reshape(logits.shape)
        # TODO: LEFT OFF HERE
        else:
            dL_dZ = caa.loss_grad(self.activation, self.hidden_state, )

        return dL_dZ
        # return dL_dZ

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