import numpy as np
import Attention.attention_head as ah

class attention_block(object):
    def __init__(self, num_heads, block_shape):
        if block_shape%num_heads != 0:
            raise Exception('Error: Attention Block input shape not divisible by number of heads.')
        
        # parameters for size
        self.num_heads = num_heads
        self.block_shape = block_shape
        self.head_output_dimension = int(block_shape/num_heads)

        # dictionary to store heads
        self.heads = {}
        for head_num in range(self.num_heads):
            head_name = f"head_{head_num}"
            self.heads[head_name] = ah.attention_head(self.block_shape, self.head_output_dimension)
        
        # weights to aggregate heads
        # initialization of weights - using only Xavier for now
        xavier_val = np.sqrt(2/(self.block_shape+self.block_shape))
        self.W_o = np.random.normal(0, xavier_val, size=(self.block_shape, self.block_shape)).astype(np.float32)
        
    def multi_head_attention(self, word_embedding):
        # calculating the scores for each head
        # TODO: Parallelize
        
        attention_scores = {}

        for head_name in self.heads.keys():
            head = self.heads[head_name]
            head_attention = head.masked_attention_score(word_embedding)
            attention_scores[head_name] = head_attention

        # concatenating scores together and running through the weights to get desired output shape
        attention_scores_list = list(attention_scores.values())
        head_attentions_concat = np.concatenate(attention_scores_list, axis=1)

        # calculating final attention scores and returning
        attention_scores_concat = head_attentions_concat @ self.W_o
        return attention_scores_concat
