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

            neuron_list: defines number of neurons in each layer, size is # layers except input
        """
        self.num_layers = len(neuron_list)
        self.layers = []
        
        # Create Neural Layers
        for neurons in neuron_list:
            layer = NeuralLayer(neurons, sigmoid)
            self.layers.append(layer)
        
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
