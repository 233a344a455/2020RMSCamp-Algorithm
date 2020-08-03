import numpy as np
import matplotlib.pyplot as plt

class SimpleNet():
    def __init__(self, loss_func, layers=[]):
        self.layers = layers
        self.loss_func = loss_func

    def predict(self, data, return_vec=False):
        """

        Args:
            data.shape = (batch, in_features)

        Returns:
            data.shape = (batch, out_features)

        """
        data = np.expand_dims(data, axis=-1) # (batch, in_features) -> (batch, in_features, 1)
        for layer in self.layers:
            data = layer.forward(data)
        if return_vec:
            return data
        else:
            return np.squeeze(data, axis=-1) # (batch, out_features, 1) -> (batch, out_features)

    def train(self, data, target):
        """

        Args:
            data.shape = (batch, in_features)
            target.shape = (batch, out_features)

        Returns:
            loss(float)

        """
        target = np.expand_dims(target, axis=-1)    # (batch, out_features) -> (batch, out_features, 1)
        loss, grad = self.loss_func(self.predict(data, return_vec=True), target)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return loss


class FullConnectedLayer:
    def __init__(self, in_features, out_features, learning_rate):
        """

        self.weight.shape = (1, out_features, in_features)
        self.weight.bias = (1, out_features, 1)

        """
        self.in_features, self.out_features = in_features, out_features
        self.learning_rate = learning_rate
        self.bias = np.random.randn(1, out_features, 1) # * 0.1
        self.weight = np.random.randn(1, out_features, in_features) # * 0.1
        self.prev_inp = None

    def forward(self, inp):
        """正向传播

        inp.shape = (batch, in_features, 1)

        """
        self.prev_inp = inp
        return self.weight @ inp + self.bias

    def backward(self, grad):
        """反向传播 The headacheing backward func

        Args:
            grad(np.array): 梯度数据(Z_curr)，shape=(batch, out_features, 1)
            prev_inp(np.array): 前一层的输出(A_prev), shape=(batch, in_features, 1)

        Returns:
            (np.array): 传给上一层的梯度

        """
        d_prev_inp = np.sum(self.weight * grad, axis=1, keepdims=True).transpose(0, 2, 1)  # prev_output的导数，用于上传梯度
        d_weight = self.prev_inp.repeat(self.out_features, axis=-1).transpose(0, 2, 1) * grad.repeat(self.in_features, axis=-1)  # weight矩阵的偏导
        d_bias = grad

        self.weight -= np.sum(d_weight, axis=0, keepdims=True) * self.learning_rate
        self.bias -= np.sum(d_bias, axis=0, keepdims=True) * self.learning_rate

        return d_prev_inp

def MSELoss(pred, target):
    """

    Args:
        pred.shape = (batch, n_featuers, 1)
        target.shape = (batch, n_featuers, 1)
    
    Returns:
        loss (float)
        gred.shape = (batch, n_featuers, 1)

    """
    loss = np.mean((pred - target) ** 2).item()
    grad = 2 * (pred - target)
    return loss, grad

class SigmoidLayer():
    def __init__(self):
        self.prev_inp = None

    def forward(self, inp):
        self.prev_inp = inp
        return 1/(1+np.exp(-inp))
    
    def backward(self, grad):
        sig = 1/(1+np.exp(-self.prev_inp))
        return grad * sig * (1 - sig)

class LeakyReLULayer():
    def __init__(self, leak):
        self.prev_inp = None
        self.leak = leak

    def forward(self, inp):
        self.prev_inp = inp
        return np.maximum(0.01*inp, inp)
    
    def backward(self, grad):
        self.prev_inp[self.prev_inp < 0] = self.leak
        self.prev_inp[self.prev_inp >= 0] = 1
        return grad * self.prev_inp
