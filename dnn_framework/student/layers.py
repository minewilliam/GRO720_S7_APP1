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
    def __init__(self, input_size, alpha=0.1):
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
        batch_mean = np.mean(x, axis=0)
        batch_variance = np.var(x, axis=0)
        epsilon = 1e-6 # Prevent division by zero

        # Update rolling mean and variance to estimate the trend of the dataset
        self._buffers["global_mean"] = (1 - alpha) * self._buffers["global_mean"] + alpha * batch_mean
        self._buffers["global_variance"] = (1 - alpha) * self._buffers["global_variance"] + alpha * batch_variance

        normalized_x = self._normalize(x, batch_mean, batch_variance, epsilon)
        return gamma * normalized_x + beta, {"beta": beta,
                                             "gamma": gamma,
                                             "x": x}

    def _forward_eval(self, x):
        beta = self._parameters["beta"]
        gamma = self._parameters["gamma"]
        mean = self._buffers["global_mean"]
        variance = self._buffers["global_variance"]
        epsilon = 1e-6 # Prevent division by zero

        normalized_x = self._normalize(x, mean, variance, epsilon)
        return gamma * normalized_x + beta, {"beta": beta,
                                             "gamma": gamma,
                                             "x": x}

    def backward(self, output_grad, cache):
        gamma = cache["gamma"]
        x = cache["x"]
        M = x.shape[0]  # Number of samples in batch
        batch_mean = np.mean(x, axis=0)
        batch_variance = np.var(x, axis=0)
        epsilon = 1e-6  # Prevent division by zero

        # Gradients:
        x_grad = (1.0 / M) * gamma * 1 / np.sqrt(batch_variance + epsilon) * (M * output_grad - np.sum(output_grad, axis=0) \
                 - (x - batch_mean) * 1 / (batch_variance + epsilon) * np.sum(output_grad * (x - batch_mean), axis=0))

        gamma_grad = np.sum(output_grad * self._normalize(x, batch_mean, batch_variance, epsilon), axis=0)
        beta_grad = np.sum(output_grad, axis=0)

        output_cache = {"gamma": gamma_grad, "beta": beta_grad}
        return x_grad, output_cache

    def _normalize(self, x, mean, variance, epsilon):
        return (x - mean) / np.sqrt(variance + epsilon)


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
