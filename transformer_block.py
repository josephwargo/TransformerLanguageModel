import numpy as np
import Attention.attention_block as ab
import feed_forward as ff
import layer_norm as ln

class transformer_block(object):
    def __init__(
              self, num_heads, input_shape
            , output_shape, activation
            , head_output_dimension
            , block_output_dimension
            , batch_size, clip_val
            , learning_rate , adam=False
            ):
        
        self.num_heads = num_heads
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.head_output_dimension = head_output_dimension
        self.block_output_dimension = block_output_dimension
        self.activation = activation
        self.batch_size = batch_size
        self.clip_val = clip_val
        self.learning_rate = learning_rate
        self.adam = adam

        # first layer norm
        self.layer_norm_1 = ln.layer_norm(input_shape)

        # self-attention
        self.self_attention = ab.attention_block(
              num_heads=num_heads, input_dimension=input_shape
            , head_output_dimension=head_output_dimension
            , block_output_dimension=block_output_dimension
        )

        # second layer norm
        self.layer_norm_2 = ln.layer_norm(block_output_dimension)

        # feed forward

        self.feed_forward_layer = ff.neuron_layer(
              input_shape=input_shape, output_shape=output_shape, activation='relu'
            , batch_size=self.batch_size, clip_val=self.clip_val, learning_rate=self.learning_rate
            , adam=self.adam
        )


    def forward_pass(self, x):
        # first layer norm and attention masked self-attention
        x_1 = self.self_attention.multi_head_attention(self.layer_norm_1.layer_norm(x))

        # residual add
        x_2 = x_1 + x

        # second layer norm & linear forward pass
        x_3 = self.feed_forward_layer.forward_pass(self.layer_norm_2.layer_norm(x_2))

        # residual add
        x_4 = x_3 + x_1

        return x_4