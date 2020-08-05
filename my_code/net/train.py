from simple_net import *
import matplotlib.pyplot as plt
import numpy as np
import random
import math

BATCH_SIZE = 32
EPOCH = 0

loss_list = []
acc_list = []
# plt.ion()

# net = SimpleNet(cross_entropy_loss, Adam(),\
#     layers=[
#         FullConnectedLayer(784, 100),
#         DropoutLayer(0.3),
#         ReLULayer(),
#         FullConnectedLayer(100, 100),
#         DropoutLayer(0.3),
#         ReLULayer(),
#         FullConnectedLayer(100, 10),
#         SoftmaxLayer()
#     ])

net = load_network('nets/net_128-128_98.1.pkl')

data, labels = read_picture.read_image_data('../mnist_data/train-images.idx3-ubyte', '../mnist_data/train-labels.idx1-ubyte')
data = np.reshape(data, (60000, 784)).astype(np.float) / 255
labels = one_hot_encode(labels, 10)

dataloader = DataLoader(data[:55000], labels[:55000], BATCH_SIZE, EPOCH)

def eval(eval_data, eval_labels):
    a = np.argmax(eval_labels, axis=1)
    b = np.argmax(net.predict(eval_data), axis=1)
    return np.mean(a==b)

for pack in dataloader:
    loss = net.train(*pack)
    
    net.optimizer.lr = 0.01 + 0.01 *  np.cos(dataloader.iter_cnt / ( 55000 / BATCH_SIZE ) * 2 * np.pi)
    if dataloader.iter_cnt % 50 == 0:
        acc = eval(data[55000:55100], labels[55000:55100])
        print("Epoch %s | Iteration %s | Loss %.4f | Acc %s" %(dataloader.epoch_cnt, dataloader.iter_cnt, loss, acc))
        loss_list.append(loss)
        acc_list.append(acc)
        # plt.plot(loss_list, color='blue')
        # plt.plot(acc_list, color='yellow')
        # plt.show()
        # plt.pause(0.01)
        # plt.clf()


# save_network(net, 'net.pkl')