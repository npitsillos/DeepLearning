import numpy as np

##################################################
# Utils
##################################################

# Input Layer
class Input():

    def __init__(self, num_inputs):
        """ Constructs an Input Layer
            
            inputs: number of inputs to the network
        """

        self.num_inputs = num_inputs + 1 # plus bias node
        self.inputs = [1] # initially we only have the bias
    
    def set_inputs(self, inputs):
        self.inputs += inputs

# Sigmoid
def sigmoid(z):
    return 1./ (1+np.exp(-z))