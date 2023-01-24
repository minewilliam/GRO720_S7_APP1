import numpy as np

from dnn_framework.loss import Loss as LossBase


class CrossEntropyLoss(LossBase):
    """
    :param x: The input tensor (e.g. model's output)
    :param target: The target tensor (e.g. true labels)
    :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
    """

    def calculate(self, x, target):
        exp_x = np.exp(x)
        prob = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        # Get the true class probability
        true_class_prob = prob[np.arange(x.shape[0]), target]

        # Calculate the cross-entropy loss
        loss = -np.mean(np.log(true_class_prob))

        input_grad = prob.copy()
        input_grad[np.arange(x.shape[0]), target] -= 1
        input_grad = input_grad / x.shape[0]

        return loss, input_grad


class MeanSquaredErrorLoss(LossBase):
    """
    :param x: The input tensor (e.g. model's output)
    :param target: The target tensor (e.g. true labels)
    :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
    """

    def calculate(self, x, target):
        loss = np.power((x - target), 2)
        input_grad = 2 * (x - target)
        return loss, input_grad


def softmax(x):
    exponential = np.exp(x)
    probabilities = exponential / np.sum(exponential)
    return probabilities
