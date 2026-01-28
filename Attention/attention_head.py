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

    def masked_attention_score(self, word_embeddings):
        # calculating q, k, and v
        self.calculate_q_k_v(word_embeddings)

        # calculating attention score
        score_matrix = self.q @ self.k.T
        scaled_score_matrix = score_matrix / self.dim_sqrt

        # determining & applying the mask
        score_mask = np.tril(np.ones(shape=scaled_score_matrix.shape))
        masked_score = np.where(score_mask, scaled_score_matrix, -1e9)

        

        #applying softmax
        softmax_masked_score = caa.softmax(masked_score)

        # print(softmax_masked_score.shape)

        # applying attention score to values to "weight" each word by its respective attention
        masked_self_attention = softmax_masked_score @ self.v

        print(masked_self_attention.shape)

        # return masked_self_attention.shape