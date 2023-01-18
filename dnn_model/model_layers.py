from dnn_framework import Layer as LayerBase
import numpy as np


class Layer(LayerBase):
    def __init__(self, input_size, layer_size):
        self._layer_size = layer_size
        self._parameters = {"weights", np.zeros([layer_size, input_size]),
                            "biases", np.zeros(layer_size)}
        self._buffers = {}
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
    def forward(self, x):
        w = self._parameters["weights"]
        b = self._parameters["biases"]
        return np.matmul(x, w.transpose()) + b, x

    def backward(self, output_grad, cache):
        w = self._parameters["weights"]
        input_grad = np.matmul(output_grad, w)
        w_grad = np.matmul(output_grad.transpose(), cache)
        b_grad = np.sum(output_grad, axis=0)
        return input_grad, w_grad, b_grad


class BatchNormalization(Layer):
    def forward(self, x):
        raise NotImplementedError()

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
