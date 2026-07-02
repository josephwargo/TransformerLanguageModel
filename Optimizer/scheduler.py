class scheduler(object):
####################################
# Initializations #
####################################
    def __init__(self, input_shape, output_shape,weight_decay=None):
        # constants
        self.beta1 = .9