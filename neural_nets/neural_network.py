import numpy as np
from neural_layer import NeuralLayer
from utils import sigmoid

##########################################################
#  Neural Network
##########################################################

"""
Implementation of a neural network
"""

class NeuralNetwork():

    def __init__(self, neuron_list):
        """
            Constructs a Neural Network

            neuron_list: defines number of neurons in each layer
                         size is # layers except input
        """
        self.num_layers = len(neuron_list)
        self.layers = []
        
        # Create Neural Layers
        for neurons in neuron_list:
            layer = NeuralLayer(neurons, sigmoid)
            self.layers.append(layer)
    
    def initialise_weights(self, num_inputs):
        """
            Initialises weights of every neuron in the 
            network

            num_inputs: number of initial inputs to the network
        """

        # Each neuron's inputs in a layer is the number of neurons in
        # the previous layer.  The number of neurons in the input layer
        # is the number of inputs.
        previous_layer_neurons = num_inputs

        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.weights = np.random.normal(0,1, previous_layer_neurons)
            previous_layer_neurons = layer.num_neurons
    
    def feed_forward(self, inputs):
        """
            Forward propagate the inputs to this network

            inputs: inputs to this Network
        """
        self.outputs = []
        
        # input neurons can be thought of as a layer that
        # just propagates the input values
        layer_outputs = inputs

        # Propagate layer outputs until final hidden layer
        for i in range(self.num_layers-1):
            layer_outputs = (self.layers[i].feed_forward(layer_outputs))
        
        # Now propagate though output layer
        self.outputs = self.layers[self.num_layers-1].feed_forward(layer_outputs)

        return self.outputs
        
    def inspect(self):
        """
            Inspect Network
        """

        print("Network")
        for i in range(len(self.layers)):
            if i != len(self.layers)-1:
                print("Hidden {}".format(i+1))
            else:
                print("Output Layer")
            self.layers[i].inspect()