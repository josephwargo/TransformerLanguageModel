import numpy as np
import Attention.attention_head as ah

class attention_block(object):
    def __init__(self, num_heads, input_dimension, head_output_dimension, block_output_dimension):
        # parameters for size
        self.num_heads = num_heads
        self.input_dimension = input_dimension
        self.head_output_dimension = head_output_dimension
        self.block_output_dimension = block_output_dimension
        # self.block_input_dimension = self.input_dimension * self.num_heads

        # dictionary to store heads
        self.heads = {}
        for head_num in range(self.num_heads):
            head_name = f"head_{head_num}"
            self.heads[head_name] = ah.attention_head(self.input_dimension, self.head_output_dimension)
        
        # weights to aggregate heads
        # initialization of weights - using only Xavier for now
        xavier_val = np.sqrt(2/(self.input_dimension+self.block_output_dimension))
        self.W_o = np.random.normal(0, xavier_val, size=(self.input_dimension, self.block_output_dimension)).astype(np.float32)

        # dict to store attention scores
        self.attention_scores = {}
        

    def multi_head_attention(self, word_embedding):
        # calculating the scores for each head
        # TODO: Parallelize
        for head_name in self.heads.keys():
            head = self.heads[head_name]
            head_attention = head.attention_score(word_embedding)
            self.attention_scores[head_name] = head_attention

        # concatenating scores together and running through the weights to get desired output shape
        attention_scores_list = list(self.attention_scores.values())
        head_attentions_concat = np.concatenate(attention_scores_list, axis=1)

        multi_head_attention = head_attentions_concat @ self.W_o

        print(multi_head_attention.shape)
        # block_attention = 0

