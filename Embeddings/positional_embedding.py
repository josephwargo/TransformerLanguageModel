import cupy as cp

class positional_embedding(object):
####################################
# Initializations #
####################################
    def __init__(
            self, max_seq_len, input_layer_shape
        ):
        
        self.max_seq_len = max_seq_len
        self.input_layer_shape = input_layer_shape

        self.embeddings = cp.random.normal(0, .02, size=(self.max_seq_len, input_layer_shape))

        self.dL_dE = cp.zeros_like(self.embeddings)

####################################
# Forward Pass #
####################################
    def forward_pass(self, x, seq_len, train=False):
        embeddings_sliced = self.embeddings[:seq_len]
        return x + embeddings_sliced

####################################
# Backward Pass #
####################################
    def backward_pass(self, dL_dY):
        # summing across batch axis so we can update pos embeddings
        # which are shape max_seq_len = maximum possible tokens x input_layer_shape
        # = shape of input token embeddings
        self.dL_dE[:self.dL_dY.shape[0]] += cp.sum(dL_dY, axis=0)


        # returning so we can pass back to input layer
        return dL_dY

    
    def update(self, learning_rate):
        # updating embeddings inplace
        # self.embeddings[:self.dL_dE.shape[0]] += -learning_rate * self.dL_dE
        self.embeddings += -learning_rate * self.dL_dE

    def clear_grad(self):
        self.dL_dE.fill(0)