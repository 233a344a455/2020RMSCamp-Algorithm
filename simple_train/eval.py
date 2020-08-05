from simple_net import *
import matplotlib.pyplot as plt
import numpy as np
import random
import math

import sys
sys.path.append("../read_picture/")
import read_picture

net = load_network('nets/net_128-128_98.1.pkl')

def eval(eval_data, eval_labels):
    a = np.argmax(eval_labels, axis=1)
    b = np.argmax(net.predict(eval_data), axis=1)
    return np.mean(a==b)


data, labels = read_picture.read_image_data('../mnist_data/t10k-images.idx3-ubyte', '../mnist_data/t10k-labels.idx1-ubyte')
data = np.reshape(data, (-1, 784)).astype(np.float) / 255
labels = one_hot_encode(labels, 10)

print(eval(data, labels))