import numpy as np
import feed_forward as ff
import costs_and_activations as caa
import Embeddings.positional_embedding as pe
import layer_norm as ln
import transformer_block as tb

# entire net
class transformer(object):
####################################
# Initial Initializations #
####################################
    def __init__(
          self
        # , embeddings, word2ind
        , input_layer_shape, input_layer_activation
        , d_model, hidden_layer_activations
        , hidden_layer_num_heads
        , output_shape
        # , output_layer_activation
        , loss_function='cross_entropy_loss'
        , learning_rate=.001, epochs=1, batch_size=8
        , adam=False, clip_val=1, debug=False
    ):
        
        self.debug = debug

        # errors
        if adam & (learning_rate>.01):
            print('Warning: Learning rate may be too high for ADAM optimizer to function properly')
       
        # variables straight from initialization
        # embeddings
        # self.embeddings = embeddings
        # self.corpus = corpus
        # self.word2ind = word2ind
        # self.word2ind_mapper = np.vectorize(word2ind.get)
        # self.embeddings_shape = embeddings.shape[1]
        # self.num_embeddings = embeddings.shape[0]

        # layer details
        self.input_layer_shape = input_layer_shape
        self.input_layer_activation = input_layer_activation
        self.d_model = d_model
        self.hidden_layer_activations = hidden_layer_activations
        self.hidden_layer_num_heads = hidden_layer_num_heads
        self.output_shape = output_shape
        # self.output_layer_activation = output_layer_activation
        
        # hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.adam = adam
        self.learning_rate = np.float32(learning_rate)
        self.loss_function = loss_function
        self.clip_val = np.float32(clip_val)
        self.activations = hidden_layer_activations# + [output_layer_activation]

        # loss
        # self.localLoss = []
        # self.loss = []
        # self.lossGradients = []

        # layers
        # self.layers = []
        
####################################
# Init Input Layer
####################################
        # TODO: make max_seq_len a thing, and update this after making d_model a thing
        self.positional_embeddings = pe.positional_embedding(max_seq_len=1024, input_layer_shape=input_layer_shape)

        self.input_layer = ff.neuron_layer(
              input_shape=self.input_layer_shape, output_shape=self.d_model
            , activation=self.input_layer_activation
            , clip_val=self.clip_val, adam=self.adam
        )

####################################
# Init Transformer Blocks
####################################
        self.transformer_layers = {} # dictionary to hold all transformer layers
        for layer_num, layer in enumerate(self.hidden_layer_activations):
            layer_activation = self.activations[layer_num]

            self.transformer_layers[f'transformer_layer_{layer_num}'] = tb.transformer_block(
                  num_heads=self.hidden_layer_num_heads, d_model=self.d_model
                , activation=layer_activation
                , clip_val=self.clip_val
                , adam=self.adam
            )

####################################
# Init Output Layer #
####################################
        output_layer_input_shape = self.d_model
        self.output_layer_norm = ln.layer_norm(output_layer_input_shape)
        self.output_layer = ff.neuron_layer(
              input_shape=output_layer_input_shape, output_shape=self.output_shape
            , activation=None # activation is none so this returns the logits, we apply the activation later for gradients
            , clip_val=self.clip_val, adam=self.adam)
        
        # TODO: revisit dictionary/embeddings
        # self.current_text = None
        # self.pad_index = self.word2ind['<PAD>']



# TODO: revisit dictionary/embeddings
    # def input_2_ind(self, text, train=True):
    #     if train:
    #         longest_sequence = max(len(sequence) for sequence in text)
    #         pad_val = '<PAD>'
    #         text = [sequence + [pad_val] * (longest_sequence - len(sequence)) for sequence in text]
    #     text = self.word2ind_mapper(text)

    #     return text

####################################
# Forward Pass #
####################################
    def forward_pass(self, x, Y=None, train=False):
        # TODO: how this changes for train vs test
        # input layer
        batch_length = x.shape[-2]
        x = self.positional_embeddings.forward_pass(x, batch_length, train)
        x = self.input_layer.forward_pass(x, train)
        # transformer blocks
        for transformer_block in self.transformer_layers.values():
            x = transformer_block.forward_pass(x, train)

        # final layer norm
        x = self.output_layer_norm.forward_pass(x, train)

        if train:
            # output layer
            logits = self.output_layer.forward_pass(x, train)

            # flattening batches
            logits_flat = logits.reshape(-1, logits.shape[-1])
            Y_flat = Y.reshape(-1)
            loss = caa.cross_entropy_loss(logits_flat, Y_flat)
        else:
            # output layer
            logits = self.output_layer.forward_pass(x[:, -1, :], train)
            loss = None

        return logits, loss
    
    def next_token_vocab_index(self, x):
        # TODO: add "temperature" so we can sample the softmax
        logits, loss = self.forward_pass(x)
        prob_dist = caa.activation('softmax', logits)
        return np.argmax(prob_dist, axis=1), prob_dist

####################################
# Backward Pass #
####################################

    def backward_pass(self, logits, Y, pad_token_ind=0):

        dL_dY = self.output_layer.backward_pass(self.learning_rate, logits=logits, Y=Y, pad_token_ind=pad_token_ind)

        dL_dY = self.output_layer_norm.backward_pass(self.learning_rate, dL_dY)

        # reversing order of transformer dict for backwards pass
        rev_transformer_layers = list(self.transformer_layers.keys())
        rev_transformer_layers.reverse()
        for layer_name in rev_transformer_layers:
            transformer_block = self.transformer_layers[layer_name]
            dL_dY = transformer_block.backward_pass(self.learning_rate, dL_dY)

        # input layer
        dL_dY = self.input_layer.backward_pass(self.learning_rate, dL_dY=dL_dY, pad_token_ind=pad_token_ind)

        self.positional_embeddings.backward_pass(self.learning_rate, dL_dY)


####################################
# Training #
####################################
    def train(self, x_batches, Y_batches, num_batches):
        for batch_num in range(num_batches):
            x_batch = x_batches[batch_num]
            Y_batch = Y_batches[batch_num]
            # forward pass
            batch_output = self.forward_pass(x_batch, Y_batch, train=True)
            logits = batch_output[0]
            loss = batch_output[1]
            
            print(f"Loss: {loss}")

            # backward pass
            dL_dY = self.backward_pass(logits=logits, Y=Y_batch)