import numpy as np
import time

class FullConnectedLayer:
    def __init__(self, in_features, out_features):
        self.in_features, self.out_features = in_features, out_features
        self.bias = np.random.randn((1, out_features)) * 0.1
        self.weight = np.random.randn((out_features, in_features)) * 0.1

    def forward(self, input):
        """正向传播
        """
        return self.weight * input + self.bias

    def backward(self, grad, prev_output, step_w, step_b):
        """反向传播 The headacheing backward func

        Args:
            grad(np.array): 梯度数据，shape=(1, out_features)
            prev_output(np.array): 前一层的输出, shape=(1, in_features)
            step_w(float): 权重步进率
            step_b(float): 截距步进率

        Returns:
            (np.array): 传给上一层的梯度

        """
        prev_output_deri = np.sum(self.weight, axis=1, keepdims=True)  # prev_output的导数，用于上传梯度
        weight_deri = prev_output.repeat(self.out_features, axis=1).T / self.in_features   # weight矩阵的偏导
        self.weight += weight_deri * grad.repeat(self.in_features, axis=1) * step_w
        self.bias += grad.sum * np.sum(step_b, axis=1, keepdims=True)

        return prev_output_deri * grad




# if __name__ == "__main__":
#     import sys
#     sys.path.append("../read_picture/")
#     import read_picture

#     train_image, train_label = read_picture.read_image_data(
#         '../mnist_data/train-images.idx3-ubyte', '../mnist_data/train-labels.idx1-ubyte')

#     train_image_vector = np.reshape(train_image, (60000, 784))

#     t0 = time.time()
#     simple_train = simple_train_one_num(train_image_vector[0:50000], train_label[0:50000], 2, 0.1, 2.55)
#     simple_train.train_learn()
#     print('Used %.4f s' %(time.time() - t0))

#     # 构造测试集
#     TEST_IMAGE_NUM = 1000
#     test_image_vector = train_image_vector[50000:50000 + TEST_IMAGE_NUM]
#     test_ans = train_label[50000:50000 + TEST_IMAGE_NUM]
#     # 计算预测
#     pred_ans = simple_train.predict(test_image_vector)

#     # 计算正确率
#     print('Acc: %.3f' %(np.sum(test_ans == pred_ans) / len(test_image_vector)))
