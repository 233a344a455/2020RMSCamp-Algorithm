import numpy as np
import time

class FullConnectedLayer:
    def __init__(self, in_features, out_features):
        """

        Args:
            in_features(float): 

        """
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
            grad(np.array): 梯度数据(Z)，shape=(1, out_features)
            prev_output(np.array): 前一层的输出(A), shape=(1, in_features)
            step_w(float): 权重步进率
            step_b(float): 截距步进率

        Returns:
            (np.array): 传给上一层的梯度

        """
        prev_output_deri = np.sum(self.weight, axis=1, keepdims=True)  # prev_output的导数，用于上传梯度
        weight_deri = self.prev_input.repeat(self.out_features, axis=1).T   # weight矩阵的偏导
        self.weight += weight_deri * grad.repeat(self.in_features, axis=1) * step_w
        self.bias += grad * np.sum(self.bias, axis=1, keepdims=True) * step_b

        return prev_output_deri * grad

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

    def backward(self, loss):
        return 2 * (self.prev_input - loss)
        
        


if __name__ == "__main__":
    fc = FullConnectedLayer(1, 1)
    fc.weight[0, 0] = 0
    fc.bias[0, 0] = 0
    print(fc.bias, fc.weight)
    # mesloss = MSELoss(10)
    # for _ in range(2):
    #     x = np.linspace(-100, 100, 50)
    #     y = x * 2.3 + np.random.randn(50)

    #     x = np.expand_dims(np.expand_dims(x, -1), -1)
    #     y = np.expand_dims(np.expand_dims(y, -1), -1)

    #     for x_, y_ in zip(x, y):
    #         t = fc.forward(x_)
    #         t = mesloss.forward(t, y_)
    #         print(t)
    #         t = mesloss.backward(t)
    #         t = fc.backward(t, 0.0001, 0.001)