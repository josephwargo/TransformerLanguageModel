import numpy as np
import feed_forward as ff
import costs_and_activations as caa
import Attention.attention_block as ab
import transformer_block as tb

# entire net
class transformer(object):
    def __init__(
          self, embeddings, word2ind
        , input_layer_shape, input_layer_activation
        , hidden_layer_shapes, hidden_layer_activations
        , output_layer_activation
        , loss_function='cross_entropy_loss'
        , learning_rate=.001, epochs=1, batch_size=8
        , adam=False, clip_val=1, debug=False
    ):
        
        self.debug = debug

        # errors
        if len(hidden_layer_shapes)!=len(hidden_layer_activations):
            raise Exception('Length of hidden_layer_shapes does not match length of hidden_layer_activations')
        if ((loss_function=='cross_entropy_loss') & (output_layer_activation!='softmax')) or ((loss_function!='cross_entropy_loss') & (output_layer_activation=='softmax')):
            raise Exception('A cost function of Cross Entropy Loss and an output layer activation of Softmax must be paired with each other')
        if adam & (learning_rate>.01):
            print('Warning: Learning rate may be too high for ADAM optimizer to function properly')
       
        # variables straight from initialization
        # embeddings
        self.embeddings = embeddings
        # self.corpus = corpus
        self.word2ind = word2ind
        self.word2ind_mapper = np.vectorize(word2ind.get)
        self.embeddings_shape = embeddings.shape[1]
        self.num_embeddings = embeddings.shape[0]

        # layer details
        self.input_layer_shape = input_layer_shape
        self.input_layer_activation = input_layer_activation
        self.hidden_layer_shapes = hidden_layer_shapes
        self.hidden_layer_activations = hidden_layer_activations
        self.output_layer_activation = output_layer_activation
        
        # hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.adam = adam
        self.learning_rate = np.float32(learning_rate)
        self.loss_function = loss_function
        self.clip_val = np.float32(clip_val)
        self.activations = hidden_layer_activations + [output_layer_activation]

        # loss
        self.localLoss = []
        self.loss = []
        self.lossGradients = []
        
        ####################################
        # Input Layer
        ####################################
        self.input_layer = ff.neuron_layer(
              input_shape=self.input_layer_shape, output_shape=self.hidden_layer_shapes[0]
            , activation=self.input_layer_activation
            , batch_size=self.batch_size, clip_val=self.clip_val, learning_rate=self.learning_rate, adam=self.adam
        )

        ####################################
        # Transformer Blocks
        ####################################
        self.transformer_layers = {} # dictionary to hold all transformer layers
        for layer_num, layer in enumerate(self.hidden_layer_shapes):
            self.transformer_layers[f'transformer_layer_{layer_num}'] = tb.transformer_block(
            # TODO: after finishing up transformer_block class, add these as parameters for the transformer object
            #   num_heads, input_dimension
            # , head_output_dimension
            # , block_output_dimension
            )

        ####################################
        # Output Layer #
        ####################################
        # TODO: Output layer
        # output_layer = ff.neuron_layer()

        self.current_text = None
        self.pad_index = self.word2ind['<PAD>']

    def input_2_ind(self, text, train=True):
        if train:
            longest_sequence = max(len(sequence) for sequence in text)
            pad_val = '<PAD>'
            text = [sequence + [pad_val] * (longest_sequence - len(sequence)) for sequence in text]
        text = self.word2ind_mapper(text)

        return text

    def train_forward_pass(self):
        pass

    def forward_pass_per_word(self, prev_layer_output, train):
        # cycling through each layer
        for layer_name in self.layers.keys():
            curr_layer = self.layers[layer_name]
            # for output layer
            if layer_name == 'output_layer':

                z = prev_layer_output @ curr_layer.layer_weights + curr_layer.bias
                logits = z

                # updating hidden state
                # curr_layer. = prev_layer_output

            # for non-output layers
            else:
                if train:
                    # adding previous layer and timestep hidden states to memory (for BPTT)
                    curr_layer.prev_layer_output_memory.append(prev_layer_output)
                    curr_layer.prevTimeStepOutputMemory.append(curr_layer.thisLayerMostRecentOutput.copy())
                
                # calculating hidden state (dot products and activation)
                # layerDotProduct = np.dot(prev_layer_output, curr_layer.layerWeights)
                layerDotProduct = prev_layer_output @ curr_layer.layerWeights
                # print(layerDotProduct.shape)
                timeDotProduct = curr_layer.thisLayerMostRecentOutput @ curr_layer.timeWeights
                # print(timeDotProduct.shape)
                # print(timeDotProduct.shape)
                z = layerDotProduct + timeDotProduct + curr_layer.bias # z = Uh + Wx + b
                hiddenState = caa.activation(curr_layer.activation, z) # activation - depending on the layer
                # print(hiddenState.shape)
                # updating hidden state and hidden state memory
                curr_layer.thisLayerMostRecentOutput = hiddenState
                if train:
                    curr_layer.thisLayerOutputMemory.append(hiddenState.copy())

                # updating so that this outputs hidden state feeds into 
                prev_layer_output = hiddenState
        return logits

    # training methods
    def forwardPass(self, text, train=True):
        # cycling through each word (timestep)
        self.current_text = self.input_2_ind(text, train)
        if train:
            numSteps = self.current_text.shape[1]
        else:
            numSteps = self.current_text.shape[0]
        for wordIndex in range(numSteps-1):
            # selecting proper input embeddings
            if train:
                inputWords = self.current_text[:,wordIndex]
            else:
                inputWords = self.current_text[wordIndex]

            prev_layer_output = self.embeddings[inputWords]
            

            # selecting index for output word
            if train:
                outputVocabIndex = self.current_text[:, wordIndex+1]
                padMask = outputVocabIndex!=self.pad_index
            else:
                outputVocabIndex = self.current_text[wordIndex+1]
            # outputVocabIndex = self.word2ind[outputWord]
            
            # iterating through each layer per word
            logits = self.forward_pass_per_word(prev_layer_output, train)

            if train:
                # storing local loss and loss gradients
                if self.loss_function == 'cross_entropy_loss':
                    loss = caa.cross_entropy_loss(outputVocabIndex[padMask], logits[padMask])
                    self.localLoss.append(loss)
                    # gradient
                    gradient = caa.softmaxLocalError(outputVocabIndex[padMask], logits[padMask])
                    fullGradient = np.zeros_like(logits)
                    fullGradient[padMask] = gradient
                    self.lossGradients.append(fullGradient) # dLdH
                else:
                    raise Exception('Unknown loss function')
        if train:
            # calculating final loss (divided over all words in the text)
            meanLoss = np.mean(self.localLoss)
            self.loss.append(meanLoss)
 
        return logits

    def backwardPassPerTimestep(self, layerLocalError, reverseKeys, timeStep):
        # iterating through layers backwards
        for layerNum, layer_name in enumerate(reverseKeys):
            layerNum = len(reverseKeys) - 1 - layerNum
            curr_layer = self.layers[layer_name]
            
            if layer_name == 'output_layer':
                # dLossdZ [stored during forward pass]
                dLossdZ = layerLocalError

                # dLossdOutputWeights [dLdW] = dZdW [previous layer hidden state output] @ dLdZ [stored during forward pass]
                prevLayerHiddenState = curr_layer.prev_layer_output_memory[timeStep]
                dLossdOutputWeights = prevLayerHiddenState.T @ dLossdZ
                # dLossdOutputWeights = dLossdOutputWeights.T
                
                # dLossdOutputBias [dLdB] = dLdZ [stored during forward pass] @ dZdB [1]
                dLossdOutputBias = np.sum(layerLocalError, axis=0)

                # dLossdPreviousHiddenLayer [dLdH] = dZdH [layerWeights] @ dLdZ [stored during forward pass]
                dLossdPrevLayerHiddenState = curr_layer.layerWeights @ dLossdZ.T
                
                # updating localError to pass back
                layerLocalError = dLossdPrevLayerHiddenState.T

                # adding gradients to list for weight updates
                curr_layer.layerWeightUpdates += dLossdOutputWeights
                curr_layer.biasUpdates += dLossdOutputBias
            
            else:
                # dLossdZ [dLdZ] = localError(dLdH [passed back from previous layer / time step], dHdZ [this layer hidden state most recent output])
                dLdH = layerLocalError+curr_layer.thisLayerTimeLocalError
                
                hiddenState = curr_layer.thisLayerOutputMemory[timeStep]
                
                dLossdZ = caa.localError(curr_layer.activation, hiddenState, dLdH)
                
                # dLossdTimeWeights [dLdWt] = dLdZ [calculated above] @ dZdWt [prevTimeStepHiddenState]
                prevTimeStepHiddenState = curr_layer.prevTimeStepOutputMemory[timeStep]
                dLossdTimeWeights = prevTimeStepHiddenState.T @ dLossdZ
                # dLossdTimeWeights = dLossdTimeWeights.T
                
                # dLossdLayerWeights [dLdWl] = dZdWl [prevLayerHiddenState] @ dLdZ [calculated above]
                prevLayerHiddenState = curr_layer.prev_layer_output_memory[timeStep]
                dLossdLayerWeights = prevLayerHiddenState.T @ dLossdZ
                # dLossdLayerWeights = dLossdLayerWeights.T
                
                # dLossdOutputBias [dLdB] = dLdZ [calculated above] @ dZdB = [1]
                dLossdOutputBias = np.sum(dLossdZ, axis=0)

                # dLossdPreviousHiddenLayer [dLdH] = dZdH [layerWeights] @ dLdZ [calculated above]
                layerLocalError = curr_layer.layerWeights @ dLossdZ.T
                layerLocalError = layerLocalError.T
                
                # dLossdPreviousTimeStep [dLdH] = dZdH [timeWeights] @ dLdZ [calculated above]
                curr_layer.thisLayerTimeLocalError = (curr_layer.timeWeights @ dLossdZ.T).T
                
                # adding gradients to list for weight updates
                curr_layer.layerWeightUpdates += dLossdLayerWeights
                curr_layer.timeWeightUpdates += dLossdTimeWeights
                curr_layer.biasUpdates += dLossdOutputBias

    # backward pass through entire text
    def backwardPass(self):
        numSteps = self.current_text.shape[1]
        # reversing layers to iterate backwards through
        reverseKeys = list(self.layers.keys())
        reverseKeys.reverse()

        # iterating through time backwards
        for reverseTimeStep in range(1, numSteps):
            timeStep = numSteps - reverseTimeStep - 1
            
            # getting proper local error that was stored during forward pass
            layerLocalError = self.lossGradients[timeStep]

            # NOT RELEVANT UNLESS WE WANT TO UPDATE EMBEDDINGS selecting proper input embeddings
            # inputWord = text[timeStep]
            # inputVocabIndex = self.word2ind[inputWord]
            # inputWordEmbedding = self.embeddings[inputVocabIndex]

            # backward pass for an individual time step
            self.backwardPassPerTimestep(layerLocalError, reverseKeys, timeStep)
        
        # updating weights and biases
        for layer_name in reverseKeys:
            curr_layer = self.layers[layer_name]            
            curr_layer.update(numSteps-1)

    
    def resetGrads(self, train=True):
    # Resetting losses and gradients in advance of forward pass
        # setting/resetting loss
        self.localLoss = [] # list to store the local loss to average at the end
        self.lossGradients = [] # list used to store loss gradients for backwards pass
                                # resetting list to empty so we only store gradients from this
                                # instance of the forward pass
        
        # setting/resetting gradients & other stored variables per layer
        for layer_name in self.layers.keys():
            curr_layer = self.layers[layer_name] # layer

            # for forward pass calculation
            # curr_layer.thisLayerMostRecentOutput = np.zeros(shape=curr_layer.thisLayerMostRecentOutput.shape).astype(np.float32)
            if train:
                curr_layer.thisLayerMostRecentOutput = np.zeros_like(curr_layer.thisLayerMostRecentOutput).astype(np.float32) # hidden state at time 0
            else:
                curr_layer.thisLayerMostRecentOutput = np.zeros(size=(curr_layer.outputShape))
            # for backward pass calculation
            curr_layer.prev_layer_output_memory = [] # memory of hidden states from the previous layer in  this timestep
            if curr_layer.rnn:
                curr_layer.prevTimeStepOutputMemory = [] # memory of hidden states from this layer in the previous timestep
            curr_layer.thisLayerOutputMemory = [] # memory of the output from this layer

            # for gradient updates
            curr_layer.thisLayerTimeLocalError = np.zeros(shape=(curr_layer.thisLayerTimeLocalError.shape)).astype(np.float32)
            if curr_layer.rnn:
                curr_layer.timeWeightUpdates = np.zeros_like(curr_layer.timeWeights).astype(np.float32)
            curr_layer.layerWeightUpdates = np.zeros_like(curr_layer.layerWeights).astype(np.float32)
            curr_layer.biasUpdates = np.zeros_like(curr_layer.bias).astype(np.float32)

    # training model by repeatedly running forward and backward passes
    def trainModel(self, corpus):
        remainder = len(corpus) % self.batch_size
        if remainder != 0:
            corpus = corpus[:-remainder]
        for count, startIndex in enumerate(range(0, len(corpus), self.batch_size)):
            stopIndex = startIndex + self.batch_size
            batch = corpus[startIndex:stopIndex]
            maxWordCount = max(len(text) for text in batch)
            print(f'Batch #{count+1} - max {maxWordCount} words')
            
            self.trainBatch(batch)
    
    def trainBatch(self, batch):
        # resetting gradients
        self.resetGrads()

        # forward pass
        _ = self.forwardPass(batch)
        modelLoss = self.loss[-1]
        print(f'Loss: {modelLoss}')
        print('********************************************')
        print()

        # backward pass
        self.backwardPass()
    
    # return predicted output for a given input
    def querySequence(self, sequence):
        # getting logits
        logits = self.forwardPass(sequence, train=False)
        
        # determining output word
        outputWordIndex = np.argmax(logits)
        outputWord = list(self.word2ind.keys())[outputWordIndex]

        return outputWord
    

    def generateOutput(self, sequence, numWords):
        self.resetGrads()
        for i in range(numWords):
            nextWord = self.querySequence(sequence)
            sequence.append(nextWord)
        
        return sequence