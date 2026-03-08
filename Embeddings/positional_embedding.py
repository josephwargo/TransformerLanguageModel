import numpy as np


class positional_embedding(object):
####################################
# Initializations #
####################################
    def __init__(
            self, max_seq_len, d_model
        ):
        
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.embeddings = np.random.normal(0, .02, size=(self.max_seq_len, d_model))

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
        # self.update(learning_rate, dL_dE)

    
    def update(self, learning_rate, dL_dE):
        self.embeddings += learning_rate * dL_dE