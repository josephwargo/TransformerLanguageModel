import numpy as np
import Attention.attention_block as ab
import feed_forward as ff
import layer_norm as ln

class transformer_block(object):
    def __init__(
              self, num_heads, block_shape
            , activation
            , batch_size, clip_val
            , learning_rate , adam=False
            ):
        
        self.num_heads = num_heads
        self.block_shape = block_shape
        self.activation = activation
        self.batch_size = batch_size
        self.clip_val = clip_val
        self.learning_rate = learning_rate
        self.adam = adam

        # first layer norm
        self.layer_norm_1 = ln.layer_norm(self.block_shape)

        # self-attention
        self.self_attention = ab.attention_block(
              num_heads=num_heads, block_shape=self.block_shape
        )

        # second layer norm
        self.layer_norm_2 = ln.layer_norm(self.block_shape)

        # feed forward
        self.feed_forward_layer = ff.neuron_layer(
              input_shape=self.block_shape, output_shape=self.block_shape, activation='relu'
            , batch_size=self.batch_size, clip_val=self.clip_val, learning_rate=self.learning_rate
            , adam=self.adam
        )

####################################
# Forward Pass #
####################################
    def forward_pass(self, x, train=False):
        # first layer norm and attention masked self-attention and residual add
        x = x + self.self_attention.multi_head_attention(self.layer_norm_1.forward_pass(x))

        # second layer norm & linear forward pass and residual add
        x = x + self.feed_forward_layer.forward_pass(self.layer_norm_2.forward_pass(x), train)

        return x

####################################
# Backward Pass #
####################################
    def backward_pass(self, dL_dY, pad_token_ind=0):
        #TODO: Feed Forward backward pass
        dL_dY, dL_dW, dL_db = self.feed_forward_layer.backward_pass(dL_dY, pad_token_ind=pad_token_ind)
        
        #TODO: Layer Norm backward pass

        # TODO: attention backward pass
        dL_dY, dL_dW, dL_db = self.self_attention.backward_pass()

        #TODO: Layer Norm backward pass

        return dL_dY, dL_dW, dL_db