import numpy as np
import pickle

class SimpleNet():
    def __init__(self, loss_func, optimizer, layers=[]):
        self.layers = layers
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.optimizer.link_layers_params(self.layers)

    def predict(self, data, train=False):
        """

        Args:
            data.shape = (batch, in_features)

        Returns:
            data.shape = (batch, out_features)

        """
        data = np.expand_dims(data, axis=-1) # (batch, in_features) -> (batch, in_features, 1)
        for layer in [l for l in self.layers if not (isinstance(l, DropoutLayer) and not train)]:
            data = layer.forward(data)
        if train:
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
        loss, grad = self.loss_func(self.predict(data, train=True), target)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        self.optimizer.step()
        return loss


class FullConnectedLayer():
    def __init__(self, in_features, out_features):
        """

        self.weight.shape = (1, out_features, in_features)
        self.weight.bias = (1, out_features, 1)

        """
        self.in_features, self.out_features = in_features, out_features
        self.bias = np.random.randn(1, out_features, 1) / np.sqrt(in_features / 2)
        self.weight = np.random.randn(1, out_features, in_features) / np.sqrt(in_features / 2)
        self.prev_inp = None
        self.d_weight = None
        self.d_bias = None

    def forward(self, inp):
        """

        inp.shape = (batch, in_features, 1)

        """
        self.prev_inp = inp
        return self.weight @ inp + self.bias

    def backward(self, grad):
        """The headaching backward func !!!

        Args:
            grad(np.array): (Z_curr) shape=(batch, out_features, 1)
            prev_inp(np.array): (A_prev) shape=(batch, in_features, 1)

        Returns:
            (np.array): shape=(batch, in_features, 1)

        """
        d_prev_inp = np.sum(self.weight * grad, axis=1, keepdims=True).transpose(0, 2, 1)  # prev_output的导数，用于上传梯度
        self.d_weight = self.prev_inp.repeat(self.out_features, axis=-1).transpose(0, 2, 1) * grad.repeat(self.in_features, axis=-1)  # weight矩阵的偏导
        self.d_bias = grad
        self.d_weight = np.mean(self.d_weight, axis=0, keepdims=True)
        self.d_bias = np.mean(self.d_bias, axis=0, keepdims=True)

        return d_prev_inp


class DropoutLayer():
    def __init__(self, dropout_rate = 0.3):
        self.dropout_rate = dropout_rate
        self.filter = None
    
    def forward(self, inp):
        self.filter = np.random.binomial(n=1, p=1-self.dropout_rate, size=inp.shape)
        return inp * self.filter
    
    def backward(self, grad):
        return grad * self.filter


# class BatchNormLayer():
#     def __init__(self, n_features, eps=1e-5, momentum=0.9):
#         self.eps = eps
#         self.momentum = momentum
#         self.n_featuers = n_featuers

#         self.gamma = np.zeros((n_featuers, 1))
#         self.beta = np.zeros((n_featuers, 1))
#         self.running_mean = 0
#         self.running_var = 0
        
#         self.out_ = None
#         self.prev_inp = None
#         self.sample_mean = None
#         self.sample_var = None
#         self.d_gamma = None
#         self.d_beta = None

#     def forward(inp, train):
#         inp = inp[...,0]
#         self.prev_inp = inp
        
#         if train:
#             self.sample_mean = np.mean(inp, axis=0)
#             self.sample_var = np.var(inp, axis=0)
#             self.out_ = (inp - self.sample_mean) / np.sqrt(self.sample_var + self.eps)
#             self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.sample_mean
#             self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.sample_var
#             out = self.gamma * self.out_ + self.beta
#         else:
#             scale = self.gamma / np.sqrt(self.running_var + self.eps)
#             out = inp * scale + (self.beta - self.running_mean * scale)

#         return out[...,np.newaxis]

#     def backward(grad):
#         grad = grad[...,0]

#         N = self.prev_inp.shape[0]
#         dout_ = self.gamma * grad
#         dvar = np.sum(dout_ * (self.prev_inp - self.sample_mean) * -0.5 * (self.sample_var + self.eps) ** -1.5, axis=0)
#         dx_ = 1 / np.sqrt(self.sample_var + self.eps)
#         dvar_ = 2 * (self.prev_inp - self.sample_mean) / N

#         di = dout_ * dx_ + dvar * dvar_
#         dmean = -1 * np.sum(di, axis=0)
#         dmean_ = np.ones_like(self.prev_inp) / N

#         dx = di + dmean * dmean_
#         self.d_gamma = np.sum(grad * self.out_, axis=0)
#         self.d_beta = np.sum(grad, axis=0)

#         return dx[...,np.newaxis]



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


def cross_entropy_loss(pred, target):
    """

    Args:
        pred.shape = (batch, n_featuers(n_kinds), 1)
        target.shape = (batch, n_featuers(n_kinds), 1)
    
    Returns:
        loss (float)
        gred.shape = (batch, n_featuers(n_kinds), 1)

    """
    loss = target * np.log(pred) + (1-target) * np.log(1-pred)
    loss[np.logical_or(np.isnan(loss), np.isinf(loss))] = 0.
    loss = -np.mean(loss)
    grad = -target/pred +(1-target)/(1-pred)
    grad[np.logical_or(np.isnan(grad), np.isinf(grad))] = 0.
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


class ReLULayer():
    def __init__(self, leak=0.01):
        self.prev_inp = None
        self.leak = leak

    def forward(self, inp):
        self.prev_inp = inp
        return np.maximum(inp, 0)
    
    def backward(self, grad):
        return grad * (self.prev_inp >= 0)


class SoftmaxLayer():
    def __init__(self):
        self.s = None
    
    def forward(self, inp):
        inp -= np.max(inp, axis=1, keepdims=True)
        self.s = np.exp(inp) / np.sum(np.exp(inp), axis=1, keepdims=True)
        # self.s[self.s <= 0] = 1e-12
        # self.s[self.s >= 1] = 1 - 1e-12
        return self.s
    
    def backward(self, grad):
        d =  self.s * np.eye(self.s.shape[1]) - self.s * self.s.transpose(0, 2, 1)
        return d @ grad


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

    def link_layers_params(self, layers):
        self.maintaining_layers = [layer for layer in layers if isinstance(layer, (FullConnectedLayer, ))]
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


# ========================== Utils ==========================


def save_network(net, path):
    with open(path, 'wb') as f:
        pickle.dump(net, f)

def load_network(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def one_hot_encode(labels, n_types):
    return np.eye(n_labels)[labels]

def one_hot_decode(pred):
    return np.argmax(pred, axis=1)


class DataLoader():
    def __init__(self, data, labels, batch_size, n_epoch):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.n_epoch = n_epoch

        self.epoch_size = len(data)
        self.rand_list = np.arange(self.epoch_size)
        np.random.shuffle(self.rand_list)
        self.iter_cnt = 0
        self.epoch_cnt = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.iter_cnt += 1
        if not self.epoch_cnt < self.n_epoch:
            raise StopIteration

        if len(self.rand_list) < self.batch_size:
            self.rand_list = np.arange(self.epoch_size)
            np.random.shuffle(self.rand_list)
            self.epoch_cnt += 1
            self.iter_cnt = 0

        idx_list, self.rand_list = self.rand_list[:self.batch_size], self.rand_list[self.batch_size:]
        return self.data[idx_list, :], self.labels[idx_list, :]