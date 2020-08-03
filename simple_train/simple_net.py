import numpy as np
import matplotlib.pyplot as plt

class FullConnectedLayer:
    def __init__(self, in_features, out_features):
        self.in_features, self.out_features = in_features, out_features
        self.bias = np.random.randn(out_features, 1) * 0.1
        self.weight = np.random.randn(out_features, in_features) * 0.1
        self.prev_input = None

    def forward(self, input):
        """正向传播
        """
        self.prev_input = input
        return np.dot(self.weight, input) + self.bias

    def backward(self, grad, step_w, step_b):
        """反向传播 The headacheing backward func

        Args:
            grad(np.array): 梯度数据(Z_curr)，shape=(1, out_features)
            prev_input(np.array): 前一层的输出(A_prev), shape=(1, in_features)
            step_w(float): 权重步进率
            step_b(float): 截距步进率

        Returns:
            (np.array): 传给上一层的梯度

        """
        # m = self.prev_input.shape[1]
        # d_weight = np.dot(grad, self.prev_input.T) / m
        # d_bias = np.sum(grad, axis=1, keepdims=True) / m
        # d_prev_inp = np.dot(self.weight.T, self.prev_input)

        d_prev_inp = np.sum(self.weight * grad, axis=0, keepdims=True).T  # prev_output的导数，用于上传梯度
        d_weight = self.prev_input.repeat(self.out_features, axis=1).T * grad.repeat(self.in_features, axis=1)  # weight矩阵的偏导
        d_bias = grad
        # print("bias=%s, grad=%s" %(d_bias, grad))

        self.weight -= d_weight * step_w
        self.bias -= d_bias * step_b

        return d_prev_inp

class MSELoss():
    def __init__(self):
        # self.in_features = in_features
        self.prev_input = None
    
    def forward(self, input, target):
        """

        Args:
            input(np.array): shape=(1, in_features)

        """
        self.prev_input = input
        return np.mean((input - target) ** 2)

    def backward(self, target):
        return 2 * (self.prev_input - target)

class SigmoidLayer():
    def __init__(self, n_features):
        self.n_features = n_features
        self.prev_input = None

    def forward(self, input):
        self.prev_input = input
        return 1/(1+np.exp(-input))
    
    def backward(self, grad):
        sig = 1/(1+np.exp(-grad))
        return self.prev_input * sig * (1 - sig)


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