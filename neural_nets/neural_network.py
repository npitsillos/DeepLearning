import numpy as np
from neural_layer import NeuralLayer
from utils import sigmoid, Input

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
                         size is number layers including input
        """
        self.num_layers = len(neuron_list)
        self.layers = []
        
        # Create Neural Layers
        i = 0
        for i in range(len(neuron_list)):
            if i == 0:
                layer = Input(neuron_list[i])
            else:
                layer = NeuralLayer(neuron_list[i], sigmoid)
            self.layers.append(layer)
    
    def initialise_weights(self):
        """
            Initialises weights of every neuron in the 
            network
        """

        # Each neuron's inputs in a layer is the number of neurons in
        # the previous layer.  The number of neurons in the input layer
        # is the number of inputs.
        previous_layer_neurons = self.layers[0].num_inputs
        i = 1
        while i < len(self.layers):
            for neuron in self.layers[i].neurons:
                neuron.weights = np.random.normal(0,1, previous_layer_neurons)
            if i < len(self.layers) - 1:
                previous_layer_neurons = self.layers[i].num_neurons + 1  # considering the bias node only until final hidden layer
            i += 1

    def set_inputs(self, inputs):
        """ Set Network inputs

            inputs: inputs to the network
        """
        self.layers[0].set_inputs(inputs)

    def feed_forward(self):
        """
            Forward propagate the inputs to this network
        """
        self.outputs = []
        
        # input neurons can be thought of as a layer that
        # just propagates the input values
        layer_outputs = self.layers[0].inputs

        # Propagate layer outputs until final hidden layer
        for i in range(1,self.num_layers-1):
            layer_outputs = self.layers[i].feed_forward(layer_outputs)
            layer_outputs.insert(0, 1) # value for the bias node
        
        print(layer_outputs)
        # Now propagate though output layer
        self.outputs = self.layers[self.num_layers-1].feed_forward(layer_outputs)

        return self.outputs
        
    def train(self, train_inputs, targets):
        """ Train neural network

            train_inputs: inputs to the network
            targets: outputs targets
        """

        self.set_inputs(train_inputs)
        self.feed_forward()

        # Calculate output neuron deltas
        output_deltas = [0] * self.layers[-1].num_neurons

        for i in range(self.layers[-1].num_neurons):
            
            # we only need to calculate dE/dz since this will be used for the hidden neurons also
            output_deltas[i] = self.layers[-1].neurons[i].error_wrt_net_input(targets)

    def inspect(self):
        """
            Inspect Network
        """

        print("Network")
        for i in range(1,len(self.layers)):
            if i != len(self.layers)-1:
                print("Hidden {}".format(i))
            else:
                print("Output Layer")
            self.layers[i].inspect()