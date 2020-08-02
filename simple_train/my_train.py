import numpy as np
import matplotlib.pyplot as plt
import time

class FullConnectedLayer:
    def __init__(self, in_features, out_features):
        """

        Args:
            in_features(float): 

        """
        self.in_features, self.out_features = in_features, out_features
        self.bias = np.random.randn(out_features, 1)
        self.weight = np.random.randn(out_features, in_features)
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
            prev_output(np.array): 前一层的输出(A_prev), shape=(1, in_features)
            step_w(float): 权重步进率
            step_b(float): 截距步进率

        Returns:
            (np.array): 传给上一层的梯度

        """


        # m = self.prev_input.shape[1]
        # d_weight = np.dot(grad, self.prev_input.T) / m
        # d_bias = np.sum(grad, axis=1, keepdims=True) / m
        # d_prev_inp = np.dot(self.weight.T, self.prev_input)

        d_prev_inp = np.sum(self.weight, axis=1, keepdims=True)  # prev_output的导数，用于上传梯度
        d_weight = self.prev_input.repeat(self.out_features, axis=1).T * grad.repeat(self.in_features, axis=1)  # weight矩阵的偏导
        d_bias = grad * self.bias

        self.weight -= d_weight * step_w
        self.bias -= d_bias * step_b

        return d_prev_inp * grad

class MSELoss():
    def __init__(self, in_features):
        self.in_features = in_features
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
        
        


if __name__ == "__main__":
    fc1 = FullConnectedLayer(1, 5)
    fc2 = FullConnectedLayer(5, 1)
    # fc.weight[0, 0] = 0
    # fc.bias[0, 0] = 0
    print(fc1.bias, fc1.weight)
    mesloss = MSELoss(1)
    loss = []
    for _ in range(3000):
        x = np.linspace(-5, 5, 50)
        y =  -1.28 * x + np.random.randn(50)

        x = np.expand_dims(np.expand_dims(x, -1), -1)
        y = np.expand_dims(np.expand_dims(y, -1), -1)


        for i, (x_, y_) in enumerate(zip(x, y)):
            t = fc1.forward(x_)
            t = fc2.forward(t)
            t = mesloss.forward(t, y_)
            loss.append(t)
            t = mesloss.backward(t)
            t = fc2.backward(t, 1e-7, 1e-7)
            t = fc1.backward(t, 1e-7, 1e-7)


        x = np.linspace(-5, 5, 50)
        y =  -1.28 * x

        y_ = []
        x_ = np.expand_dims(np.expand_dims(x, -1), -1)
        for xx in x_:
            # print('a %s' %fc2.forward(fc1.forward(xx)))
            y_.append(fc2.forward(fc1.forward(xx)).flatten().item())
        
        # print(y_)
        plt.plot(x, y, color='green')
        plt.plot(x, y_, color='red')
        # plt.plot(loss)
        plt.pause(0.1)
        plt.clf()

# plt.show()