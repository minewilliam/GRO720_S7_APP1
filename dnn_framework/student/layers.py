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
        self._parameters = {"w": np.zeros([layer_size, input_size]),
                            "b": np.zeros(layer_size)}

    def forward(self, x):
        weights = self._parameters["w"]
        biases = self._parameters["b"]
        return np.matmul(x, weights.transpose()) + biases, x

    def backward(self, output_grad, cache):
        weights = self._parameters["w"]
        input_grad = np.matmul(output_grad, weights)
        weights_grad = np.matmul(output_grad.transpose(), cache)
        biases_grad = np.sum(output_grad, axis=0)
        return input_grad, {"w": weights_grad, "b": biases_grad}


class BatchNormalization(Layer):
    def __init__(self, input_size, alpha=1.0):
        Layer.__init__(self)
        self._alpha = alpha
        # the gamma vector is initialised with ones
        self._parameters = {"gamma": np.ones(input_size),
                            "beta": np.zeros(input_size)}
        # Arbitrary values so that we do not start processing with 0 or 1,
        # which are unlikely learning results
        self._buffers = {"global_mean": np.full(input_size, 0.0),
                         "global_variance": np.full(input_size, 0.0)}

    def forward(self, x):
        if self._is_training:
            return self._forward_train(x)
        return self._forward_eval(x)

    def _forward_train(self, x):
        alpha = self._alpha
        beta = self._parameters["beta"]
        gamma = self._parameters["gamma"]
        mean = np.mean(x, axis=0)
        variance = np.var(x, axis=0)

        # Update rolling mean and variance
        self._buffers["global_mean"]       = (1 - alpha) * self._buffers["global_mean"]     + alpha * mean
        self._buffers["global_variance"]   = (1 - alpha) * self._buffers["global_variance"] + alpha * variance

        normalized_x = (x - mean) / np.sqrt(variance + 1e-6)
        return gamma * normalized_x + beta, {"beta": beta, "gamma": gamma}

    def _forward_eval(self, x):
        beta = self._parameters["beta"]
        gamma = self._parameters["gamma"]
        mean = self._buffers["global_mean"]
        variance = self._buffers["global_variance"]

        normalized_x = (x - mean) / np.sqrt(variance + 1e-6)
        return gamma * normalized_x + beta, {"beta": beta, "gamma": gamma}


    def backward(self, output_grad, cache):
        raise NotImplementedError()


class ReLU(Layer):
    def forward(self, x):
        return x * (x >= 0), x

    def backward(self, output_grad, cache):
        return output_grad * (cache >= 0), None


class Sigmoid(Layer):
    def forward(self, x):
        return 1 / (1 + np.exp(-1 * x)), x

    def backward(self, output_grad, cache):
        y = self.forward(cache)[0]
        return y * (1 - y) * output_grad, None
