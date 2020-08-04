from simple_net import *
import matplotlib.pyplot as plt
import numpy as np
import random
import math

import sys
sys.path.append("../read_picture/")
import read_picture

BATCH_SIZE = 10
EPOCH = 1

data, labels = read_picture.read_image_data('../mnist_data/train-images.idx3-ubyte', '../mnist_data/train-labels.idx1-ubyte')
data = np.reshape(data, (60000, 784)).astype(np.float128) / 255
labels = np.eye(10)[labels]

net = SimpleNet(cross_entropy_loss, Adam(),\
    layers=[
        FullConnectedLayer(784, 10),
        SigmoidLayer(),
        FullConnectedLayer(10, 10),
        SigmoidLayer(),
        FullConnectedLayer(10, 10),
        SigmoidLayer()
    ])

for epoch in range(EPOCH):
    rand_list = random.sample(range(60000), k=60000)
    z = 0
    while True:
        try:
            idx_list, rand_list = rand_list[:BATCH_SIZE], rand_list[BATCH_SIZE:]
        except IndexError:
            break
        
        loss = net.train(data[idx_list, :], labels[idx_list, :])
        if math.isnan(loss):
            exit()

        z += 1
        if z % 10 == 0:
            print(loss)