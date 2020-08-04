from simple_net import *
import matplotlib.pyplot as plt
import numpy as np
import random
import math

import sys
sys.path.append("../read_picture/")
import read_picture

np.seterr(all='raise')

BATCH_SIZE = 100
EPOCH = 5

data, labels = read_picture.read_image_data('../mnist_data/train-images.idx3-ubyte', '../mnist_data/train-labels.idx1-ubyte')
data = np.reshape(data, (60000, 784)).astype(np.float128) / 255
labels = one_hot_encode(labels, 10)

# net = SimpleNet(cross_entropy_loss, Adam(),\
#     layers=[
#         FullConnectedLayer(784, 32),
#         SigmoidLayer(),
#         FullConnectedLayer(32, 16),
#         SigmoidLayer(),
#         FullConnectedLayer(16, 10),
#         SigmoidLayer()
#     ])
net = load_network('net1.pkl')

loss_list = []
plt.ion()

dataloader = DataLoader(data, labels, BATCH_SIZE, EPOCH)

for pack in dataloader:
    loss = net.train(*pack)
    
    if dataloader.iter_cnt % 20 == 0:
        print("Epoch %s | Iteration %s | Loss %.4f" %(dataloader.epoch_cnt, dataloader.iter_cnt, loss))
        # loss_list.append(loss)
        # plt.plot(loss_list)
        # plt.show()
        # plt.pause(0.1)
        # plt.clf()

save_network(net, 'net1.pkl')