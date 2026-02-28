import numpy as np
import Attention.attention_head as ah

class attention_block(object):
    def __init__(self, num_heads, d_model):
        if d_model%num_heads != 0:
            raise Exception('Error: Attention Block input shape not divisible by number of heads.')
        
        # parameters for size
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_output_dimension = int(d_model/num_heads)

        # dictionary to store heads
        # self.heads = {}
        # for head_num in range(self.num_heads):
        #     head_name = f"head_{head_num}"
        #     self.heads[head_name] = ah.attention_head(self.d_model, self.head_output_dimension)
        self.head = ah.attention_head(self.num_heads, self.d_model)
        
        # weights to aggregate heads
        # initialization of weights - using only Xavier for now
        xavier_val = np.sqrt(2/(self.d_model+self.d_model))
        self.W_o = np.random.normal(0, xavier_val, size=(self.d_model, self.d_model)).astype(np.float32) # does not matter whether in_dim or out_dim is first as both are d_model

        # hidden state
        self.hidden_state = None
        self.prev_layer_hidden_state = None
        
    def forward_pass(self, x, train=False):
        # calculating the scores for each head
        
        head_attention = self.head.forward_pass(x, train)

        # concatenating scores together and running through the weights to get desired output shape
        # concatenating by reshaping from (batch, seq_len, num_heads, head_shape) to (batch, seq_len, d_model)
        print(head_attention.shape)
        # need to transpose so that the dimensions I am trying to combine ()
        head_attention_transposed = head_attention.transpose(0,2,1,3)
        head_attentions_concat = head_attention_transposed.reshape(x.shape[0], x.shape[1], self.d_model)

        # calculating final attention scores and returning
        attention_scores_concat = head_attentions_concat @ self.W_o

        if train:
            self.hidden_state = attention_scores_concat
            self.prev_layer_hidden_state = x

        return attention_scores_concat

    def backward_pass(self, dL_dY):
        # dL_dZ = dL_dY because there is no activation
        dL_dZ_flat = dL_dY.reshape(-1, dL_dY.shape[-1])

        # dL_dW for W_o
        prev_layer_hidden_state_flat = self.prev_layer_hidden_state.reshape(-1, self.prev_layer_hidden_state.shape[-1])
        dL_dW = dL_dZ_flat.T @ prev_layer_hidden_state_flat
        
        # dL_dY - dL_dZ = dL_dY because there is no activation
        dL_dY = dL_dY @ self.W_o.T

        # reshaping dL_dY to be 4d (batch, seq_len, num_heads, head_shape) so we can do matrix multiplication with k, q, and v in the backwards pass of the attention heads
        new_dL_dY_shape_1 = (dL_dY.shape[0], self.num_heads, dL_dY.shape[1], self.head_output_dimension)
        dL_dY_4d = dL_dY.reshape(new_dL_dY_shape_1)
        
        self.head.backward_pass(dL_dY_4d)

        print("here")
        return dL_dY, dL_dW