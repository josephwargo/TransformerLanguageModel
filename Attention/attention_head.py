import numpy as np
import costs_and_activations as caa

class attention_head(object):
    def __init__(self, num_heads, d_model):
        
        # storing dimensions
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dimension = int(self.d_model / self.num_heads)
        self.dim_sqrt = np.sqrt(self.head_dimension)

        # initialization of weights - using only Xavier for now
        self.dim_for_xavier = self.d_model + self.head_dimension
        xavier_val = np.sqrt(2/(self.dim_for_xavier))

        # instead of having separate heads, initiating each weight matrix with the shape (d_model, d_model), as this is mathematically equivalent to having separate heads
        self.W_q = np.random.normal(0, xavier_val, size=(self.d_model, self.d_model)).astype(np.float32) # query weights
        self.W_k = np.random.normal(0, xavier_val, size=(self.d_model, self.d_model)).astype(np.float32) # key weights
        self.W_v = np.random.normal(0, xavier_val, size=(self.d_model, self.d_model)).astype(np.float32) # value weights

        # initialization of q, k, and v vectors
        self.q = None
        self.k = None
        self.v = None

        # for storing during train forward pass
        self.softmax_masked_score = None

    def calculate_q_k_v(self, x):
        # reshaping the output to be (batch, seq_len, num_heads, head_shape) as this is needed to calculate the attention score for different heads
        # will later reshape to (batch, seq_len, d_model) when we "concatenate" the separate heads in the attention block
        self.q = (x @ self.W_q).reshape(x.shape[0], self.num_heads, x.shape[1], self.head_dimension)
        self.k = (x @ self.W_k).reshape(x.shape[0], self.num_heads, x.shape[1], self.head_dimension)
        self.v = (x @ self.W_v).reshape(x.shape[0], self.num_heads, x.shape[1], self.head_dimension)

        # return q, k, v

    def forward_pass(self, x, train=False):
        # calculating q, k, and v
        self.calculate_q_k_v(x)

        k_T = self.k.transpose((0,1,3,2))

        # calculating attention score & scaling
        score_matrix = self.q @ k_T
        scaled_score_matrix = score_matrix / self.dim_sqrt
        
        # determining & applying the mask
        score_mask = np.tril(np.ones(shape=scaled_score_matrix.shape))
        masked_score = np.where(score_mask, scaled_score_matrix, -1e9)

        #applying softmax
        softmax_masked_score = caa.softmax(masked_score)


        # applying attention score to values to "weight" each word by its respective attention
        masked_self_attention = softmax_masked_score @ self.v
        if train:
            self.softmax_masked_score = softmax_masked_score
        return masked_self_attention
    
    def backward_pass(self, dL_dY):
        dL_dv = self.softmax_masked_score @ dL_dY
        # gradient w.r.t. softmax_masked_scores
        print("here")
        print(self.v.shape)
        print(dL_dY.shape)
        dL_dY = self.v @ dL_dY
        print(dL_dY.shape)