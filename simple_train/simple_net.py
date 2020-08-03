import numpy as np
import matplotlib.pyplot as plt

class SimpleNet():
    def __init__(self, loss_func, layers=[]):
        self.layers = layers
        self.loss_func = loss_func

    def predict(self, data):
        for layer in self.layers:
            data = layer.forward(data)
        return data

    def train(self, data, target):
        loss, grad = self.loss_func(self.predict(data), target)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad


class FullConnectedLayer:
    def __init__(self, in_features, out_features, learning_rate):
        self.in_features, self.out_features = in_features, out_features
        self.learning_rate = learning_rate
        self.bias = np.random.randn(out_features, 1) * 0.1
        self.weight = np.random.randn(out_features, in_features) * 0.1
        self.prev_inp = None

    def forward(self, inp):
        """正向传播
        """
        self.prev_inp = inp
        return np.dot(self.weight, inp) + self.bias

    def backward(self, grad):
        """反向传播 The headacheing backward func

        Args:
            grad(np.array): 梯度数据(Z_curr)，shape=(1, out_features)
            prev_inp(np.array): 前一层的输出(A_prev), shape=(1, in_features)

        Returns:
            (np.array): 传给上一层的梯度

        """
        # m = self.prev_inp.shape[1]
        # d_weight = np.dot(grad, self.prev_inp.T) / m
        # d_bias = np.sum(grad, axis=1, keepdims=True) / m
        # d_prev_inp = np.dot(self.weight.T, self.prev_inp)

        d_prev_inp = np.sum(self.weight * grad, axis=0, keepdims=True).T  # prev_output的导数，用于上传梯度
        d_weight = self.prev_inp.repeat(self.out_features, axis=1).T * grad.repeat(self.in_features, axis=1)  # weight矩阵的偏导
        d_bias = grad
        # print("bias=%s, grad=%s" %(d_bias, grad))

        self.weight -= d_weight * self.learning_rate
        self.bias -= d_bias * self.learning_rate

        return d_prev_inp

def MSELoss(pred, target):
    loss = np.mean((pred - target) ** 2)
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
