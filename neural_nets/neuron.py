import numpy as np

##########################################################
#  Neuron
##########################################################

"""
Implementation of a single neuron
"""

class Neuron:
    
    def __init__ (self, act, bias):
        """
            Constructs a Neuron

            act: activation function
            bias: bias of neuron
        """
        self.bias = bias
        self.weights = []
        self.act = act

    def feed_forward(self, inputs):
        """
            Forward propagate the inputs to this neuron

            inputs: vector of inputs to Neuron
        """
        self.inputs = inputs
        dot_prod_res = np.dot(inputs,self.weights) + self.bias
        self.outputs = self.act(dot_prod_res)
        return self.act(dot_prod_res)