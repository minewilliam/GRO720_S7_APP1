from dnn_framework.loss import Loss as LossBase


class CrossEntropyLoss(LossBase):
    """
    :param x: The input tensor
    :param target: The target tensor
    :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
    """
    def calculate(self, x, target):
        raise NotImplementedError()

class MeanSquareErrorLoss(LossBase):
    """
    :param x: The input tensor
    :param target: The target tensor
    :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
    """
    def calculate(self, x, target):
        raise NotImplementedError()