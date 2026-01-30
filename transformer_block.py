import numpy as np
import Attention.attention_block as ab
import feed_forward as ff
import layer_norm as ln

class transformer_block(object):
    def __init__(
            self, num_heads, input_dimension
            , head_output_dimension
            , block_output_dimension
            ):

        # first layer norm
        # TODO: update shapes to be hyperparameters
        self.layer_norm_1 = ln.layer_norm(input_dimension)

        # self-attention
        self.self_attention = ab.attention_block(
              num_heads=num_heads, input_dimension=input_dimension
            , head_output_dimension=head_output_dimension
            , block_output_dimension=block_output_dimension
        )

        # residual add
        # TODO: remember to execute this during the forward pass

        # second layer norm
        self.layer_norm_2 = ln.layer_norm(block_output_dimension)

        # feed forward
        # TODO: update shapes to be hyperparameters
        feed_forward_layer = ff.neuron_layer(
            input_shape=100, output_shape=100, activation='relu', batch_size=self.batch_size
            , clip_val=self.clip_val, learning_rate=self.learning_rate, adam=self.adam
        )

        # residual add
        # TODO: remember to execute this during the forward pass