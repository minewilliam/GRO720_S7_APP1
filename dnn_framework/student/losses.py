import numpy as np

from dnn_framework.loss import Loss as LossBase


class CrossEntropyLoss(LossBase):
    """
    :param x: The input tensor (e.g. model's output)
    :param target: The target tensor (e.g. true labels)
    :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
    """

    def calculate(self, x, target):
        loss = (-1 * target) * np.log(x) - (1 - target) * np.log(1 - x)
        input_grad = -1 * (target / x) + ((1 - target) / (1 - x))
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

def softmax():
    pass