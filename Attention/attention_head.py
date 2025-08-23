import numpy as np
import costs_and_activations as caa

class attention_head(object):
    def __init__(self, input_dimension, head_dimension):
        
        # storing dimensions
        self.input_dimension = input_dimension
        self.head_dimension = head_dimension
        self.dim_sqrt = np.sqrt(head_dimension)


        # initialization of weights - using only Xavier for now
        xavier_val = np.sqrt(2/(self.input_dimension+self.head_dimension))

        self.W_q = np.random.normal(0, xavier_val, size=(self.input_dimension, self.head_dimension)).astype(np.float32) # query weights
        self.W_k = np.random.normal(0, xavier_val, size=(self.input_dimension, self.head_dimension)).astype(np.float32) # key weights
        self.W_v = np.random.normal(0, xavier_val, size=(self.input_dimension, self.head_dimension)).astype(np.float32) # value weights

        # initialization of q, k, and v vectors
        self.q = None
        self.k = None
        self.v = None

    def calculate_q_k_v(self, word_embeddings):
        self.q = word_embeddings @ self.W_q
        self.k = word_embeddings @ self.W_k
        self.v = word_embeddings @ self.W_v

    def attention_score(self, word_embeddings):
        # calculating q, k, and v
        self.calculate_q_k_v(word_embeddings)

        # calculating attention score
        score_matrix = self.q @ self.k.T
        scaled_score_matrix = score_matrix / self.dim_sqrt
        softmax_scaled_score_matrix = caa.softmax(scaled_score_matrix)

        # applying attention score to values to "weight" each word by its respective attention
        self_attention = softmax_scaled_score_matrix @ self.v

        return self_attention