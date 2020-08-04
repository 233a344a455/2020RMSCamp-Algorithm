import numpy as np
import matplotlib.pyplot as plt

class SimpleNet():
    def __init__(self, loss_func, optimizer, layers=[]):
        self.layers = layers
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.optimizer.link_layers_params(self.layers)

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
        self.optimizer.step()
        return loss


class FullConnectedLayer:
    def __init__(self, in_features, out_features):
        """

        self.weight.shape = (1, out_features, in_features)
        self.weight.bias = (1, out_features, 1)

        """
        self.in_features, self.out_features = in_features, out_features
        self.bias = np.random.randn(1, out_features, 1) # * 0.1
        self.weight = np.random.randn(1, out_features, in_features) # * 0.1
        self.prev_inp = None
        self.d_weight = None
        self.d_bias = None

    def forward(self, inp):
        """正向传播

        inp.shape = (batch, in_features, 1)

        """
        self.prev_inp = inp
        return self.weight @ inp + self.bias

    def backward(self, grad):
        """反向传播 
        The headaching backward func !!!

        Args:
            grad(np.array): 梯度数据(Z_curr)，shape=(batch, out_features, 1)
            prev_inp(np.array): 前一层的输出(A_prev), shape=(batch, in_features, 1)

        Returns:
            (np.array): 传给上一层的梯度

        """
        d_prev_inp = np.sum(self.weight * grad, axis=1, keepdims=True).transpose(0, 2, 1)  # prev_output的导数，用于上传梯度
        self.d_weight = self.prev_inp.repeat(self.out_features, axis=-1).transpose(0, 2, 1) * grad.repeat(self.in_features, axis=-1)  # weight矩阵的偏导
        self.d_bias = grad
        self.d_weight = np.mean(self.d_weight, axis=0, keepdims=True)
        self.d_bias = np.mean(self.d_bias, axis=0, keepdims=True)

        return d_prev_inp


# ========================== Loss Functions ==========================


def MSE_loss(pred, target):
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


# ======================== Activation Layers ========================


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
    def __init__(self, leak=0.01):
        self.prev_inp = None
        self.leak = leak

    def forward(self, inp):
        self.prev_inp = inp
        return np.maximum(0.01*inp, inp)
    
    def backward(self, grad):
        self.prev_inp[self.prev_inp < 0] = self.leak
        self.prev_inp[self.prev_inp >= 0] = 1
        return grad * self.prev_inp


# ========================== Optimizers ==========================


class BGD():
    def __init__(self, lr):
        self.lr = lr
    
    def link_layers_params(self, layers):
        self.maintaining_layers = [layer for layer in layers if isinstance(layer, FullConnectedLayer)]

    def step(self):
        for layer in self.maintaining_layers:
            layer.weight -= layer.d_weight * self.lr
            layer.bias -= layer.d_bias * self.lr


class Adam():
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epislon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epislon = epislon
        # self.t = None
        # self.m = None
        # self.v = None

    def link_layers_params(self, layers):
        self.maintaining_layers = [layer for layer in layers if isinstance(layer, FullConnectedLayer)]
        self.t = 0
        self.m= [0.] * (2 * len(self.maintaining_layers))
        self.v = [0.] * (2 * len(self.maintaining_layers))
    
    def step(self):
        self.t += 1
        for i, layer in enumerate(self.maintaining_layers):
            layer.weight -= self.calculate_delta(layer.d_weight, i*2)
            layer.bias -= self.calculate_delta(layer.d_bias, i*2 + 1)

    def calculate_delta(self, grad, idx):
        self.m[idx] = self.beta1 * self.m[idx] + (1 - self.beta1) * grad
        self.v[idx] = self.beta2 * self.v[idx] + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m[idx] / (1 - self.beta1 ** self.t)
        v_hat = self.v[idx] / (1 - self.beta2 ** self.t)
        return self.lr * m_hat / (v_hat ** 0.5 + self.epislon)
