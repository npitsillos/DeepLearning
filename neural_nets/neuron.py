import numpy as np

##########################################################
#  Neuron
##########################################################


"""
Implementation of a single neuron
"""

class Neuron:
    
    def __init__ (self, act, bias):
        self.bias = bias
        self.weights = []
        self.act = act

    def feed_forward(self, inputs):
        self.inputs = inputs
        dot_prod_res = np.dot(inputs,self.weights) + self.bias
        self.outputs = self.act(dot_prod_res)
        return self.act(dot_prod_res)

def sigmoid(z):
    return 1./(1+ np.exp(-z))

neuron = Neuron(bias=np.random.normal(0, 1), act=sigmoid)
print(neuron.bias)
"Lecture wheather demonstration"
inputs = [ 0, 1 ]
num_inputs = len(inputs)

neuron.weights.append(2)
neuron.weights.append(4)

neuron.feed_forward(inputs)
print(neuron.outputs)