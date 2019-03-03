import numpy as np

##########################################################
#  Neuron
##########################################################

"""
Implementation of a single neuron
"""

class Neuron:
    
    def __init__ (self, act):
        """
            Constructs a Neuron

            act: activation function
        """
        self.weights = []
        self.act = act

    def feed_forward(self, inputs):
        """
            Forward propagate the inputs to this neuron

            inputs: vector of inputs to Neuron
        """
        self.inputs = inputs
        dot_prod_res = np.dot(inputs,self.weights)
        self.output = self.act(dot_prod_res)
        return self.act(dot_prod_res)

    def error_wrt_net_input(self, target):
        """ Calculation of dE/dz = dE/dy * dy/dz """
        return self.error_wrt_output(target) * self.output_wrt_input()

    def error_wrt_output(self, target):
        """ Calculation of dE/dy """
        return -(target - self.output)
        
    def output_wrt_input(self):
        """ Calculation of dy/dz """
        return self.output*(1-self.output)

    def input_wrt_weight(self, weight_index):
        """ Calculation of dz/dw """
        return self.inputs[weight_index]