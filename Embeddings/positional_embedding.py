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

        # determining batch size so we can scale the gradients - but need to make sure there are actual batches first!
        if len(dL_dY.shape) < 3:
            batch_size = 1
        else:
            batch_size = dL_dY.shape[0]

        self.update(learning_rate, dL_dE, batch_size)

    
    def update(self, learning_rate, dL_dE, batch_size):
        pad_to_add = self.max_seq_len - dL_dE.shape[1]
        dL_dE_padded = np.sum(
              np.pad(dL_dE, ((0,0), (0,pad_to_add), (0,0)), mode='constant', constant_values=0)
            , axis=0
        )
        
        self.embeddings += -learning_rate * (dL_dE_padded / batch_size)