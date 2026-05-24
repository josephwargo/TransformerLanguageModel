import numpy as np
import feed_forward as ff
import costs_and_activations as caa
import Embeddings.positional_embedding as pe
import layer_norm as ln
import transformer_block as tb
import json

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
        self.input_layer = ff.neuron_layer(
              input_shape=self.input_layer_shape, output_shape=self.d_model
            , activation=self.input_layer_activation
            , clip_val=self.clip_val, adam=self.adam
        )
        self.positional_embeddings = pe.positional_embedding(max_seq_len=1024, input_layer_shape=self.d_model)

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
            , clip_val=self.clip_val, is_output_layer=True, adam=self.adam)

####################################
# Forward Pass #
####################################
    def forward_pass(self, x, Y=None, train=False):
        # input layer
        # batch_length = x.shape[-1]
        seq_len = x.shape[-2]
        x = self.input_layer.forward_pass(x, train)
        x = self.positional_embeddings.forward_pass(x, seq_len, train)
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
        dL_dY = self.positional_embeddings.backward_pass(self.learning_rate, dL_dY)
        dL_dY = self.input_layer.backward_pass(self.learning_rate, dL_dY=dL_dY, pad_token_ind=pad_token_ind)

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
            
            print(f"Batch: {batch_num}")
            print(f"Loss: {loss}")
            print("")
            # backward pass
            dL_dY = self.backward_pass(logits=logits, Y=Y_batch)


####################################
# Misc Functions #
####################################
    def get_model_dict(self):
        model_dict = {}

        # input layer
        model_dict['input_layer_weights'] = self.input_layer.layer_weights
        model_dict['input_layer_biases'] = self.input_layer.bias

        # pos embeddings
        model_dict['positional_embeddings'] = self.positional_embeddings.embeddings

        # transformer layers
        for layer_name, block in self.transformer_layers.items():
            # layer norm 1
            model_dict[f'{layer_name}_layer_norm_1_gamma'] = block.layer_norm_1.gamma
            model_dict[f'{layer_name}_layer_norm_1_beta'] = block.layer_norm_1.beta

            # attention block
            model_dict[f'{layer_name}_attention_block_W_q'] = block.self_attention.head.W_q
            model_dict[f'{layer_name}_attention_block_W_k'] = block.self_attention.head.W_k
            model_dict[f'{layer_name}_attention_block_W_v'] = block.self_attention.head.W_v
            model_dict[f'{layer_name}_attention_block_W_o'] = block.self_attention.W_o

            # layer norm 2
            model_dict[f'{layer_name}_layer_norm_2_gamma'] = block.layer_norm_2.gamma
            model_dict[f'{layer_name}_layer_norm_2_beta'] = block.layer_norm_2.beta

            # feed forward
            model_dict[f'{layer_name}_feed_forward_weights'] = block.feed_forward_layer.layer_weights
            model_dict[f'{layer_name}_feed_forward_biases'] = block.feed_forward_layer.bias
        
        # output layer norm
        model_dict['output_layer_norm_gamma'] = self.output_layer_norm.gamma
        model_dict['output_layer_norm_beta'] = self.output_layer_norm.beta

        # output layer
        model_dict['output_layer_weights'] = self.output_layer.layer_weights
        model_dict['output_layer_biases'] = self.output_layer.bias

        config = {
        "input_layer_shape": self.input_layer_shape,
        'input_layer_activation': self.input_layer_activation,
        "d_model": self.d_model,
        "hidden_layer_activations": self.hidden_layer_activations,
        "hidden_layer_num_heads": self.hidden_layer_num_heads,
        "output_shape": self.output_shape
        }

        return model_dict, config

    def save_model(self, file_path):
        model_dict, config = self.get_model_dict()

        with open(f'{file_path}/config.json', 'w') as f:
            json.dump(config, f)

        np.savez_compressed(f'{file_path}/model.npz', **model_dict)

    def load_model(self, file_path):
        with open(f'{file_path}/config.json', 'r') as f:
            config = json.load(f)
        
        # initiating model with stored configs
        # model = bt.transformer(**config)

        # initiating weights and biases based on what we have stored
        model_dict = np.load(f'{file_path}/model.npz')


        # input layer
        self.input_layer.layer_weights = model_dict['input_layer_weights']
        self.input_layer.bias = model_dict['input_layer_biases']

        # pos embeddings
        self.positional_embeddings.embeddings = model_dict['positional_embeddings']

        # transformer layers
        for layer_name, block in self.transformer_layers.items():
            # layer norm 1
            block.layer_norm_1.gamma = model_dict[f'{layer_name}_layer_norm_1_gamma']
            block.layer_norm_1.beta = model_dict[f'{layer_name}_layer_norm_1_beta']

            # attention block
            block.self_attention.head.W_q = model_dict[f'{layer_name}_attention_block_W_q']
            block.self_attention.head.W_k = model_dict[f'{layer_name}_attention_block_W_k']
            block.self_attention.head.W_v = model_dict[f'{layer_name}_attention_block_W_v']
            block.self_attention.W_o = model_dict[f'{layer_name}_attention_block_W_o']

            # layer norm 2
            block.layer_norm_2.gamma = model_dict[f'{layer_name}_layer_norm_2_gamma']
            block.layer_norm_2.beta = model_dict[f'{layer_name}_layer_norm_2_beta']

            # feed forward
            block.feed_forward_layer.layer_weights = model_dict[f'{layer_name}_feed_forward_weights']
            block.feed_forward_layer.bias = model_dict[f'{layer_name}_feed_forward_biases']
        
        # output layer norm
        self.output_layer_norm.gamma = model_dict['output_layer_norm_gamma']
        self.output_layer_norm.beta = model_dict['output_layer_norm_beta']

        # output layer
        self.output_layer.layer_weights = model_dict['output_layer_weights']
        self.output_layer.bias = model_dict['output_layer_biases']