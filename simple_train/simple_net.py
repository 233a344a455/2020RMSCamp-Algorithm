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

if __name__ == "__main__":
    fc1 = FullConnectedLayer(1, 10)
    fc2 = FullConnectedLayer(10, 1)
    sig = SigmoidLayer(10)
    mesloss = MSELoss()

    loss = []
    i = 0
    for epoch in range(3000):
        x = np.random.randn(50) * 3
        y =  2 * x #np.sin(x*0.2)# + np.random.randn(50)

        x = np.expand_dims(np.expand_dims(x, -1), -1)
        y = np.expand_dims(np.expand_dims(y, -1), -1)

        for i, (x_, y_) in enumerate(zip(x, y)):
            t = fc1.forward(x_)
            t = sig.forward(t)
            t = fc2.forward(t)
            t = mesloss.forward(t, y_)
            loss.append(t)
            t = mesloss.backward(y_)
            t = fc2.backward(t, 0.05, 0.01)
            t = sig.backward(t)
            t = fc1.backward(t, 0.05, 0.01)


        x = np.linspace(-5, 5, 50)
        y =  2 * x#np.sin(x*0.2)

        y_ = []
        x_ = np.expand_dims(np.expand_dims(x, -1), -1)
        for xx in x_:
            # print('a %s' %fc2.forward(fc1.forward(xx)))
            y_.append(fc2.forward(sig.forward(fc1.forward(xx))).flatten().item())
            # y_.append(fc1.forward(xx).flatten().item())

        # print(y_)
        plt.plot(x, y, color='green')
        plt.plot(x, y_, color='red')
        # if epoch % 5 == 0:
        plt.pause(0.1)
        plt.clf()
        # plt.plot(loss)

# plt.show()