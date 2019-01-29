import numpy as np
from neuron import Neuron

##########################################################
#  Neural Layer
##########################################################

"""
Implementation of a layer of neurons
"""

class NeuralLayer():

    def __init__(self, num_neurons, act):
        """ Constructs a Neural Layer
            
            num_neurons: number of Neurons width of layer
            act: layer activation function
        """
        self.num_neurons = num_neurons
        self.neurons = []

        for i in range(self.num_neurons):
            neuron = Neuron(act=act)
            self.neurons.append(neuron)

    def feed_forward(self, inputs):
        """
            Forward propagate the inputs to this Neural Lyaer Neurons

            inputs: inputs to the Neural Layer
        """
        self.outputs = [] 
        for neuron in self.neurons:
            self.outputs.append(neuron.feed_forward(inputs))
        return self.outputs

    def calculate_total_error(self, targets):
        """ Calculates total error of network

            targets: list of target of neurons
        """
        diff = targets - outputs
        diff = diff ** 2
        return np.sum(diff)/2
    
    def inspect(self):
        """
            Inpsect this layer
        """
        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            print("\tNeuron_{}".format(i))
            for j in range(len(neuron.weights)):
                print("\t\tWeight_{}: {}".format(j, neuron.weights[j]))