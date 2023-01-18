from dnn_framework import Layer as LayerBase
import numpy as np


class Layer(LayerBase):
    def __init__(self):
        self._buffers = {}
        self._parameters = {}
        LayerBase.__init__(self)

    def get_parameters(self):
        return self._parameters

    def get_buffers(self):
        return self._buffers

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        raise NotImplementedError()


class FullyConnectedLayer(Layer):
    def __init__(self, input_size, layer_size):
        Layer.__init__(self)
        self._layer_size = layer_size
        self._parameters = {"weights": np.zeros([layer_size, input_size]),
                            "biases": np.zeros(layer_size)}

    def forward(self, x):
        weights = self._parameters["weights"]
        biases = self._parameters["biases"]
        return np.matmul(x, weights.transpose()) + biases, x

    def backward(self, output_grad, cache):
        weights = self._parameters["weights"]
        input_grad = np.matmul(output_grad, weights)
        weights_grad = np.matmul(output_grad.transpose(), cache)
        biases_grad = np.sum(output_grad, axis=0)
        return input_grad, weights_grad, biases_grad


class BatchNormalization(Layer):
    def __init__(self, input_size, training_size, alpha):
        Layer.__init__(self, input_size)
        self._training_size = training_size
        self._alpha = alpha
        # the gamma vector is initialised with ones
        self._parameters = {"gamma": np.ones(input_size),
                            "beta": np.zeros(input_size)}
        # Arbitrary values so that we do not start processing with 0 or 1,
        # which are unlikely learning results
        self._buffers = {"mean": 0.2,
                         "variance": 0.8}

    def forward(self, x):
        alpha = self._alpha
        beta = self._parameters["beta"]
        gamma = self._parameters["gamma"]

        # Update dataset mean and variance
        self._buffers["mean"]       = (1 - alpha) * self._buffers["mean"]     + alpha * np.mean(x)
        self._buffers["variance"]   = (1 - alpha) * self._buffers["variance"] + alpha * np.var(x)

        normalized_x = (x - np.mean(x)) / np.sqrt(np.power(np.var(x), 2) + 1e-6)
        return gamma * normalized_x + beta


    def backward(self, output_grad, cache):
        raise NotImplementedError()


class ReLU(Layer):
    def forward(self, x):
        return x * (x >= 0), x

    def backward(self, output_grad, cache):
        raise 1.0 * (cache >= 0)


class Sigmoid(Layer):
    def forward(self, x):
        return np.power(1 + np.exp(-1 * x), -1), x

    def backward(self, output_grad, cache):
        return self.forward(cache) * (1 - self.forward(cache)) * output_grad
