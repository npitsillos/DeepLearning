import random
from neural_network import NeuralNetwork
from keras.datasets import mnist

net = NeuralNetwork([2,2,1])
net.initialise_weights()
net.inspect()

training_sets = [
                [[0,0], [0]],
                [[0,1], [1]],
                [[1,0], [1]],
                [[1,1], [1]],
            ]

for i in range(1000):
    inputs, targets = random.choice(training_sets)
    net.train(inputs, targets)
    net.clear_inputs()

net.set_inputs([0,0])
print(net.feed_forward())