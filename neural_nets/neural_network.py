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

    LEARNING_RATE = 0.5

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

    def clear_inputs(self):
        """ Re-initialises Input layer inputs
        """
        self.layers[0].clear_inputs()

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
        
        # Now propagate though output layer
        return self.layers[self.num_layers-1].feed_forward(layer_outputs)
        
    def train(self, train_inputs, targets):
        """ Train neural network

            train_inputs: inputs to the network
            targets: output targets
        """

        self.set_inputs(train_inputs)
        self.feed_forward()

        # Calculate output neuron deltas
        output_deltas = [0] * self.layers[-1].num_neurons

        for i in range(self.layers[-1].num_neurons):
            # we only need to calculate dE/dz since this will be used for the hidden neurons also
            # avoid recalculating
            output_deltas[i] = self.layers[-1].neurons[i].error_wrt_net_input(targets)

        # Store previous layer deltas since we iterating over all hidden layers
        previous_layer_deltas = output_deltas
        # Hidden layer neuron deltas
        hidden_layers_neuron_deltas = []
        # Iterate backwards over last hidden to first hidden layer
        j = len(self.layers[1:-1])
        while j > 0:
            # Hidden layer delta calculation depends on what is propagated back from all neurons
            # connected to current neuron.
            hidden_deltas = [0] * self.layers[j].num_neurons
            for i in range(len(hidden_deltas)):
                error_wrt_hidden_neuron_output = 0
                # Iterate over previous layer deltas
                for o in range(len(previous_layer_deltas)):
                    # Error is dE_nextlayer/dout_hidden = dE_nextlayerneuron/dnet_nextlayerneuron * dnet_nextlayer/dout_nextlayer
                    # We calculate error of the next layer with respect to this hidden layer neuron
                    error_wrt_hidden_neuron_output += previous_layer_deltas[o] * self.layers[j+1].neurons[o].weights[i]

                hidden_deltas[i] = error_wrt_hidden_neuron_output
                
            previous_layer_deltas = hidden_deltas
            for i in range(len(hidden_deltas)):
                hidden_deltas[i] = hidden_deltas[i]  * self.layers[j].neurons[i].output_wrt_input()

            hidden_layers_neuron_deltas.append(hidden_deltas)
            j -= 1

        # Until this point we have dE/dz for all output neurons
        # For every hidden layer neuron dE/dnet
        # Get final value of all output neuron weight delta and adjust
        for o in range(self.layers[-1].num_neurons):
            for w_index in range(len(self.layers[-1].neurons[o].weights)):
                w_delta = output_deltas[o] * self.layers[-1].neurons[o].input_wrt_weight(w_index)

                # Update weight
                self.layers[-1].neurons[o].weights[w_index] -= self.LEARNING_RATE * w_delta
        
        # Update weights for all hidden layers neurons
        j = len(self.layers[1:-1])
        while j > 0:
            for h in range(len(self.layers[j].neurons)):
                for w_h in range(len(self.layers[j].neurons[h].weights)):
                    w_delta = hidden_layers_neuron_deltas[j-1][h] * self.layers[j].neurons[h].input_wrt_weight(w_h)
                    self.layers[j].neurons[h].weights[w_h] -= self.LEARNING_RATE * w_delta
            j -= 1

    def calculate_total_error(self, training_sets):
        """ Calculates total error of network

            training_sets : list of training sets with inputs and target
        """
        total = 0
        for i in range(len(training_sets)):
            inputs, targets = training_sets[i]
            self.set_inputs(inputs)
            outputs = self.feed_forward()
            for o in range(len(outputs)):
                total += 0.5 * (targets[o] - outputs[o]) ** 2
            self.clear_inputs()
        return total
            
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