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

    def forward(self, input):
        """正向传播
        """
        return np.dot(self.weight, input) + self.bias

    def backward(self, grad, prev_output, step_w, step_b):
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
        weight_deri = prev_output.repeat(self.out_features, axis=1).T   # weight矩阵的偏导
        self.weight += weight_deri * grad.repeat(self.in_features, axis=1) * step_w
        self.bias += grad * np.sum(self.bias, axis=1, keepdims=True) * step_b

        return prev_output_deri * grad




if __name__ == "__main__":
    x = np.linspace(-1, 1, 50)
    y = x * 2.3 + np.random.randn(50)

    fc = FullConnectedLayer(1, 1)
    for x_, y_ in zip(x, y):
        np.expand_dims