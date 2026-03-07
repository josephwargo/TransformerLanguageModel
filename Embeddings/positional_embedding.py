import numpy as np


class positional_embedding(object):
    def __init__(
            self, max_seq_len, d_model
        ):
        
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.embeddings = np.random.normal(0, .02, size=(self.max_seq_len, d_model))

    def forward_pass(self, x, seq_len, train=False):
        embeddings_sliced = self.embeddings[:seq_len]
        return x + embeddings_sliced
    
    def backward_pass(self, dL_dY):
        dL_dembeddings = dL_dY

        # TODO: Update embeddings based on gradient