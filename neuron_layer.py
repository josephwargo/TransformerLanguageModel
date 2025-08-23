import numpy as np
class neuron_layer(object):
    def __init__(self, prevLayerShape, outputShape, activation, batchSize, clipVal, learningRate, adam=False):

        # layer info
        self.prevLayerShape = prevLayerShape
        self.outputShape = outputShape
        layerXavier = np.sqrt(2/(self.prevLayerShape+self.outputShape))
        self.layerWeights = np.random.normal(0,layerXavier, size=(self.prevLayerShape,self.outputShape)).astype(np.float32)

        self.bias = np.zeros(shape=(outputShape)).astype(np.float32)

        # activation
        self.activation = activation

        # hyperparameters
        self.batchSize = batchSize
        self.clipVal = clipVal
        self.learningRate = learningRate

        # adam
        self.adam = adam
        if adam:
            # constants
            self.beta1 = .9
            self.beta2 = .999
            self.epsilon = 1e-8
            self.t = 1
            # arrays to store 
            self.mdLayerWeights = np.zeros(shape=(self.prevLayerShape,self.outputShape)).astype(np.float32)
            self.vdLayerWeights = np.zeros(shape=(self.prevLayerShape,self.outputShape)).astype(np.float32)

            self.mdBias = np.zeros(shape=(outputShape)).astype(np.float32)         
            self.vdBias = np.zeros(shape=(outputShape)).astype(np.float32)
    
    def update(self, numSteps):

        # normalizing
        self.layerWeightUpdates /= numSteps
        self.biasUpdates /= numSteps

        # clipping
        np.clip(self.layerWeightUpdates, -self.clipVal, self.clipVal, out=self.layerWeightUpdates)
        np.clip(self.biasUpdates, -self.clipVal, self.clipVal, out=self.biasUpdates)

        # adam
        if self.adam:
            self.updateAdam()
        
        else:
            self.layerWeights += -self.learningRate*self.layerWeightUpdates
            self.bias += -self.learningRate*self.biasUpdates


    def updateAdam(self):
        
        # doing ^t on beta1 and beta2 once per step
        b1T = self.beta1**self.t
        b2T = self.beta2**self.t
        
        # layer weights
        # momentum stored
        self.mdLayerWeights *= self.beta1
        self.mdLayerWeights += (1-self.beta1)*self.layerWeightUpdates
        # RMSProp stored
        self.vdLayerWeights *= self.beta2
        self.vdLayerWeights += (1-self.beta2)*(self.layerWeightUpdates**2)
        # updates (incl. corrections and Adam)
        self.layerWeights += -self.learningRate * (self.mdLayerWeights / (1-b1T) / (np.sqrt(self.vdLayerWeights / (1-b2T))+self.epsilon))

        # bias
        # momentum stored
        self.mdBias *= self.beta1
        self.mdBias += (1-self.beta1)*self.biasUpdates
        # RMSProp stored
        self.vdBias *= self.beta2
        self.vdBias += (1-self.beta2)*(self.biasUpdates**2)
        # update (incl. corrections and Adam)
        self.bias += -self.learningRate * (self.mdBias / (1-b1T) / (np.sqrt(self.vdBias / (1-b2T))+self.epsilon))

        self.t+=1 # increment