import numpy as np 
from sklearn.utils import shuffle, resample
from sklearn.datasets import load_boston

class Layer(object):
    def __init__(self, inbound_layers=[]):
        self.inbound_layers  = inbound_layers
        self.outbound_layers = [] 
        self.value = None
        # Gradients: Keys: Layer inputs, Values: Layer Partial Derviatives w/respect to input
        self.gradients = {}
        for layer in inbound_layers:
            layer.outbound_layers.append(self)
    
    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

class Input(Layer):
    def __init__(self):
        # an input layer has no inbound layers
        Layer.__init__(self)
    
    def forward(self):
        pass 

    def backward(self):
        # an input layer has no inputs to gradient: so gradient is zero for this layer  
        self.gradients = {self: 0}
        for n in self.outbound_layers:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1

class Linear(Layer):
    def __init__(self, inbound_layer, weights, bias):
        Layer.__init__(self, [inbound_layer, weights, bias])
    
    def forward(self):
        inputs  = self.inbound_layers[0].value
        weights = self.inbound_layers[1].value
        bias    = self.inbound_layers[2].value
        self.value = np.dot(inputs, weights) + bias

    def backward(self):
        # Initialize partial gradientfor each of the inbound_layers.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_layers}
        # Cycle through and sum the outputs. The gradient will change depending depending on each output
        for n in self.outbound_layers:
            # Get the partial of the cost with respect to this layer.
            grad_cost = n.gradients[self]
            # Set the partial of the loss with respect to this layer's inputs.
            self.gradients[self.inbound_layers[0]] += np.dot(grad_cost, self.inbound_layers[1].value.T)
            # Set the partial of the loss with respect to this layer's weights.
            self.gradients[self.inbound_layers[1]] += np.dot(self.inbound_layers[0].value.T, grad_cost)
            # Set the partial of the loss with respect to this layer's bias.
            self.gradients[self.inbound_layers[2]] += np.sum(grad_cost, axis=0, keepdims=False)


class Sigmoid(Layer):
    def __init__(self, layer):
        Layer.__init__(self, [layer])
    
    def _sigmoid(self, x):
        # used in both forward and backward functionality 
        return 1. / (1. + np.exp(-x))

    def forward(self):
        input_value = self.inbound_layers[0].value
        self.value  = self._sigmoid(input_value)

    def backward(self):
        # Initialize the gradients to 0.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_layers}
        # Cycle through and sum the outputs. The gradient will change depending depending on each output
        for n in self.outbound_layers:
            # Get the partial of the cost with respect to this layer.
            grad_cost = n.gradients[self]
            sigmoid   = self.value
            # summation of gradient via derivative formula of sigmoid
            # let grad_cost = dcost/dsigmoid
            # let sigmoid*(1-sigmoid) = dsigmoid/dx
            self.gradients[self.inbound_layers[0]] += sigmoid * (1 - sigmoid) * grad_cost


class MSE(Layer):
    def __init__(self, y, a):
        # Mean Squared Error Cost Function 
        Layer.__init__(self, [y,a])
    
    def forward(self): 
        # calculate mean squared error 
        y = self.inbound_layers[0].value.reshape(-1, 1)
        a = self.inbound_layers[1].value.reshape(-1, 1) 
        self.m = (self.inbound_layers[0].value.shape[0])
        self.diff  = y-a
        #self.value = 1./self.m * np.sum(self.diff**2)
        self.value = np.mean(self.diff**2)

    def backward(self):
        # calculates the gradient of the cost as theh final layer of the network
        self.gradients[self.inbound_layers[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_layers[1]] = (-2 / self.m) * self.diff


def topological_sort(feed_dict):
    # sort the layers in topological order using Kahn' algorithm 
    # feed_dict: keys are input layer, value is repective value feed into layer
    # return a list of sorted layers

    input_layers = [n for n in feed_dict.keys()]
    G = {}
    layers = [n for n in input_layers]
    while len(layers) > 0:
        n = layers.pop(0)
        if n not in G: G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_layers:
            if m not in G: G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            layers.append(m)

    L = []
    S = set(input_layers)
    while len(S) > 0:
        n = S.pop()
        if isinstance(n, Input): n.value = feed_dict[n]
        L.append(n)
        for m in n.outbound_layers:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0: S.add(m)
    return L


def forward_and_backward(graph):
    for n in graph: n.forward()
    for n in graph[::-1]: n.backward()

def gradient_descent_update(x, gradx, learning_rate):
    # gradient is initially in direction of steepest ascent, so subtract x from it
    return (x - (learning_rate * gradx))

def sgd_update(train_params, learning_rate=1e-2):
    # iterate through each parameter layers (input layers) to update the value
    # equivalent to: current value - learning_rate*gradient(cost)
    for t in train_params:
            partial = t.gradients[t]
            t.value -= learning_rate * partial


if __name__ == '__main__':
    # Extract data 
    data = load_boston()
    features = data['data']
    target   = data['target']
    # Preprocess the data 
    features_normed = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    n_obs, n_features = features_normed.shape
    n_hidden, n_epochs, batch_size = (10, 10, 11)
    steps_per_epoch = n_obs // batch_size
    # Configure Weights and Bias 
    W1_, b1_ = (np.random.randn(n_features, n_hidden), np.zeros(n_hidden))
    W2_, b2_ = (np.random.randn(n_hidden, 1), np.zeros(1))
    # Create Input layers 
    X, y   = (Input(), Input())
    W1, b1 = (Input(), Input()) 
    W2, b2 = (Input(), Input())  
    trainables = [W1, b1, W2, b2]
    # Via composition of functions: MSE(Linear(Sigmoid(Linear(X, W1, b1)), W2, b2), y).  
    l1 = Linear(X, W1, b1)
    s1 = Sigmoid(l1)
    l2 = Linear(s1, W2, b2)
    cost = MSE(y, l2)

    feed_dict = {
        X: features_normed, y: target,
        W1: W1_, b1: b1_,
        W2: W2_, b2: b2_
    }
    graph = topological_sort(feed_dict)
    for epoch in range(n_epochs):
        loss = 0
        for step in range(steps_per_epoch):
            # Randomly sample a batch of examples
            X_batch, y_batch = resample(features_normed, target, n_samples=batch_size)        
            # Reset value of X and y Input Layers
            X.value, y.value = (X_batch, y_batch)
            # Perform forward and backward propagation
            forward_and_backward(graph)
            # update weights and bias
            sgd_update(trainables)
            # keep count of running loss per each epoch
            loss += graph[-1].value
        print("Epoch: {}, Loss: {:.3f}".format(epoch+1, loss/steps_per_epoch))
