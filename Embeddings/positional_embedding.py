import numpy as np


class positional_embedding(object):
####################################
# Initializations #
####################################
    def __init__(
            self, max_seq_len, input_layer_shape
        ):
        
        self.max_seq_len = max_seq_len
        self.input_layer_shape = input_layer_shape

        self.embeddings = np.random.normal(0, .02, size=(self.max_seq_len, input_layer_shape))

####################################
# Forward Pass #
####################################
    def forward_pass(self, x, seq_len, train=False):
        embeddings_sliced = self.embeddings[:seq_len]
        return x + embeddings_sliced

####################################
# Backward Pass #
####################################
    def backward_pass(self, learning_rate, dL_dY):
        dL_dE = dL_dY

        self.update(learning_rate, dL_dE)

        # returning so we can pass back to input layer
        return dL_dY

    
    def update(self, learning_rate, dL_dE):
        # summing across batch axis so we can update pos embeddings
        # which are shape max_seq_len = maximum possible tokens x input_layer_shape
        # = shape of input token embeddings
        dL_dE_summed = np.sum(dL_dE, axis=0)
 
        # updating embeddings inplace
        self.embeddings[:dL_dE_summed.shape[0]] += -learning_rate * dL_dE_summed