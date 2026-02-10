import numpy as np

# functions to compute costs and gradients of costs
def MSE(y, y_pred):
    return .5*np.mean((y-y_pred)**2)
def MSEgradient(y, y_pred):
    return (y_pred-y) / y.size

# no gradient function because this will only be paired with softmax, and they have a joint gradient function
def cross_entropy_loss(logits, Y_ind):
    # determining maxes to normalize
    max_logits = logits.max(axis=1, keepdims=True)

    # using log sum exp trick
    norm_exp = np.exp(logits - max_logits)
    sum_norm_exp = norm_exp.sum(axis=1, keepdims=True)
    log_sum_exp = max_logits + np.log(sum_norm_exp+1e-12)

    # geting the log probability
    true_val_logits = logits[np.arange(logits.shape[0]), Y_ind][:, None]
    log_prob = log_sum_exp - true_val_logits

    return log_prob.mean()

# functions to compute activation and gradient of activations
def sigmoid(x):
    return 1 / (1 + np.exp(-(x)))
def sigmoidGradient(y):
    return y*(1-y)

def relu(x):
    return np.maximum(0, x)
def reluGradient(y):
    return (y>0)*1

def tanH(x):
    numerator = np.exp(x) - np.exp(-x)
    denominator = np.exp(x) + np.exp(-x)
    return numerator/denominator
def tanHGradient(y):
    return 1 - y**2

# no gradient function because this will only be paired with Cross Entropy Loss,
# and they have a joint gradient function
def softmax(logits):
    if len(logits.shape)>1:
        normalization = np.max(logits, axis=1, keepdims=True)
        numerator = np.exp(logits - normalization)
        denominator = np.sum(numerator, axis=1, keepdims=True)
    else:
        normalization = np.max(logits)
        numerator = np.exp(logits - normalization)
        denominator = np.sum(numerator)
    return numerator / (denominator+1e-8)

# special gradient for softmax & cross entropy loss
def softmax_cross_entropy_grad(logits, Y_ind):
    prob_dist = softmax(logits)
    prob_dist[np.arange(prob_dist.shape[0]), Y_ind] -= 1.0
    return prob_dist / logits.shape[0]
    # return prob

# choosing activation
def activation(activation_name, z):
    if activation_name == 'relu':
        return relu(z)
    elif activation_name =='sigmoid':
        return sigmoid(z)
    elif activation_name == 'tanH':
        return tanH(z)
    elif activation_name == 'softmax':
        return softmax(z)
    else:
        raise Exception('Unknown activation function')

# choosing gradient
def loss_grad(activation_name, hidden_state, dL_dY):
    if activation_name == 'relu':
        dY_dZ = reluGradient(hidden_state)
        dL_dZ = dL_dY * dY_dZ
    elif activation_name == 'sigmoid':
        dY_dZ = sigmoidGradient(hidden_state)
        dL_dZ = dL_dY * dY_dZ
    elif activation_name == 'tanH':
        dY_dZ = tanHGradient(hidden_state)
        dL_dZ = dL_dY * dY_dZ
    # special case of softmax & cross entropy loss
    # TBD
    return dL_dZ