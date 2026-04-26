import numpy as np
import costs_and_activations as caa


class neuron_layer(object):
####################################
# Initializations #
####################################
    def __init__(self, input_shape, output_shape, activation,
                 clip_val, is_output_layer=False, adam=False):

        # layer info 
        self.input_shape = input_shape # length of vector that will be inputted into the weights
        self.output_shape = output_shape # desired length of weight output (and therefore layer output)
        layer_xavier = np.sqrt(2/(self.input_shape+self.output_shape)) # value used to determine optimal
        # shape is (out_dim, in_dim) for hardware optimization. this is NOT untuitive
        self.layer_weights = np.random.normal(0,layer_xavier, size=(self.output_shape, self.input_shape)).astype(np.float32)

        self.bias = np.zeros(shape=(output_shape)).astype(np.float32) # initializing bias as 0s - no xavier here

        # activation
        self.activation = activation # activation function that will be applied to the output of the weights

        # hyperparameters
        self.clip_val = clip_val # used to set the upper and lower bounds for what we will let weight updates hit

        # hidden state
        self.hidden_state = None
        self.prev_layer_hidden_state = None

        # type of layer
        self.is_output_layer = is_output_layer

        # adam
        # TODO: implement
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
        # if it is an output layer - no activation
        if self.is_output_layer:
            hidden_state = x @ self.layer_weights.T + self.bias
        # an other layer - yes activation
        else:
            hidden_state = caa.activation(self.activation, x @ self.layer_weights.T + self.bias)
        
        if train:
            self.hidden_state = hidden_state
            self.prev_layer_hidden_state = x
        return hidden_state

####################################
# Backward Pass #
####################################
    def backward_pass(self, learning_rate, dL_dY=None, logits=None, Y=None, pad_token_ind=0):

        if self.is_output_layer:
            
            # flattening and masking logits and Y
            Y_flat = Y.reshape(-1)
            mask = (Y_flat != pad_token_ind)
            Y_flat_masked = Y_flat[mask]
            logits_flat = logits.reshape(-1, logits.shape[-1])
            logits_flat_masked = logits_flat[mask]
            
            # grad for non-padded
            print(logits_flat_masked.shape)
            dL_dZ_active = caa.softmax_cross_entropy_grad(logits_flat_masked, Y_flat_masked)
            # reshaping to pre-flattened shape for dL_dZ
            dL_dZ_flat = np.zeros_like(logits_flat)
            dL_dZ_flat[mask] = dL_dZ_active
            dL_dZ = dL_dZ_flat.reshape(logits.shape)
        
        # any other layer
        else:
            # calculating and flatting dL_dZ
            dL_dZ = caa.loss_grad(self.activation, self.hidden_state, dL_dY)
            dL_dZ_flat = dL_dZ.reshape(-1, dL_dZ.shape[-1])
            
        # dL_dx
        dL_dx = dL_dZ @ self.layer_weights

        # dL_dW - requries flattening the previous layer hidden state and using the flattened dL_dZ
        prev_layer_hidden_state_flat = self.prev_layer_hidden_state.reshape(-1, self.prev_layer_hidden_state.shape[-1])
        dL_dW = dL_dZ_flat.T @ prev_layer_hidden_state_flat
        
        # dL_db
        dL_db = np.sum(dL_dZ, axis=(0,1))

        # determining batch size so we can scale the gradients - but need to make sure there are actual batches first!
        if len(dL_dx.shape) < 3:
            batch_size = 1
        else:
            batch_size = dL_dx.shape[0]
        
        self.update(learning_rate, dL_dW, dL_db, batch_size)

        return dL_dx

    def update(self, learning_rate, dL_dW, dL_db, batch_size):
        # clipping
        # np.clip(self.layer_weight_updates, -self.clip_val, self.clip_val, out=self.layer_weight_updates)
        # np.clip(self.bias_updates, -self.clip_val, self.clip_val, out=self.bias_updates)

        # adam
        if self.adam:
            self.update_adam()
        
        else:
            self.layer_weights += -learning_rate * (dL_dW / batch_size)
            self.bias += -learning_rate * (dL_db / batch_size)

    # TODO: update update_adam - this is leftover from RNN
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