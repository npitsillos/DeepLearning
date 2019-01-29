from neural_network import NeuralNetwork

net = NeuralNetwork([2,2,1])
net.initialise_weights()
net.set_inputs([0,1])
net.inspect()
print(net.feed_forward())