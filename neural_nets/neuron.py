import numpy as np

##########################################################
#  Neuron
##########################################################


"""
Implementation of a single neuron
"""

class Neuron:
    
    def __init__ (self, act_fun, bias):
        self.bias = bias
        self.weights = []
        self.act_fun = act_fun

    def feed_forward(self, inputs):
        dot_prod_res = np.dot(inputs,self.weights) + self.bias
        return self.act_fun(dot_prod_res)
    
    """
        Partial derivative calculation wrt to each weight is given by dE/dw.
        Applying the chain rule twice results in:
            dE/dw = dE/dz  * dz/dw -> once
            dE/dw = dE/dy * dy/dz * -> twice
            where: E is the cost function
                   y is the unit's output y = w^Tx + b
                   z is the unit's input for the specific w
                   w is the weight
    """