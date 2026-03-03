import numpy as np
import Attention.attention_block as ab
import feed_forward as ff
import layer_norm as ln

class transformer_block(object):
    def __init__(
              self, num_heads, d_model
            , activation
            , batch_size, clip_val
            , learning_rate , adam=False
            ):
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.activation = activation
        self.batch_size = batch_size
        self.clip_val = clip_val
        self.learning_rate = learning_rate
        self.adam = adam

        # first layer norm
        self.layer_norm_1 = ln.layer_norm(self.d_model)

        # self-attention
        self.self_attention = ab.attention_block(
              num_heads=num_heads, d_model=self.d_model
        )

        # second layer norm
        self.layer_norm_2 = ln.layer_norm(self.d_model)

        # feed forward
        self.feed_forward_layer = ff.neuron_layer(
              input_shape=self.d_model, output_shape=self.d_model, activation='relu'
            , batch_size=self.batch_size, clip_val=self.clip_val, learning_rate=self.learning_rate
            , adam=self.adam
        )

####################################
# Forward Pass #
####################################
    def forward_pass(self, x, train=False):
        # first layer norm and attention masked self-attention and residual add
        residual_with_self_attention = x + self.self_attention.forward_pass(self.layer_norm_1.forward_pass(x, train), train)

        # second layer norm & linear forward pass and residual add
        transformer_block_output = residual_with_self_attention + self.feed_forward_layer.forward_pass(
            self.layer_norm_2.forward_pass(residual_with_self_attention, train), train)

        return transformer_block_output

####################################
# Backward Pass #
####################################
    def backward_pass(self, dL_dY, pad_token_ind=0):
        #TODO: Feed Forward backward pass
        dL_dY = self.feed_forward_layer.backward_pass(dL_dY, pad_token_ind=pad_token_ind)
        
        #TODO: Layer Norm backward pass
        dL_dY = self.layer_norm_2.backward_pass(dL_dY)

        # attention backward pass
        dL_dY = self.self_attention.backward_pass(dL_dY)

        #TODO: Layer Norm backward pass

        #TODO: make updates work for dL_dW, dL_db

        return dL_dY