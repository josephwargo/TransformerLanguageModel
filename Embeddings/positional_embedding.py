import numpy as np


class positional_embedding(object):
    def __init__(
            self, max_seq_len, d_model
        ):
        
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.embeddings = np.random.normal(0, .02, size=(self.max_seq_len, d_model))

    def forward_pass(self, x):
        return x + self.embeddings