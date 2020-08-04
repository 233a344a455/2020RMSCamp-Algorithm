from simple_net import *
import matplotlib.pyplot as plt
import numpy as np
import random
import math

import sys
sys.path.append("../read_picture/")
import read_picture

# np.seterr(all='raise')

BATCH_SIZE = 100
EPOCH = 5

loss_list = []
plt.ion()

net = SimpleNet(cross_entropy_loss, Adam(lr=0.008),\
    layers=[
        FullConnectedLayer(784, 32),
        SigmoidLayer(),
        FullConnectedLayer(32, 16),
        # DropoutLayer(0.3),
        SigmoidLayer(),
        FullConnectedLayer(16,10),
        # DropoutLayer(0.3),
        SigmoidLayer(),
        FullConnectedLayer(10, 10),
        SigmoidLayer()
    ])

# net = load_network('net3.pkl')

data, labels = read_picture.read_image_data('../mnist_data/train-images.idx3-ubyte', '../mnist_data/train-labels.idx1-ubyte')
data = np.reshape(data, (60000, 784)).astype(np.float) / 255
labels = one_hot_encode(labels, 10)


dataloader = DataLoader(data[:50000], labels[:50000], BATCH_SIZE, EPOCH)

for pack in dataloader:
    loss = net.train(*pack)
    
    if dataloader.iter_cnt % 20 == 0:
        print("Epoch %s | Iteration %s | Loss %.4f" %(dataloader.epoch_cnt, dataloader.iter_cnt, loss))
        # loss_list.append(loss)
        # plt.plot(loss_list)
        # plt.show()
        # plt.pause(0.1)
        # plt.clf()

a = np.argmax(labels[50000:], axis=1)
b = np.argmax(net.predict(data[50000:]), axis=1)
print(np.mean(a==b))

save_network(net, 'net3.pkl')