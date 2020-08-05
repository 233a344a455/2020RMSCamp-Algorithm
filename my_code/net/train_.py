from simple_net import *
import numpy as np
import random
import math
import pickle

BATCH_SIZE = 64
EPOCH = 0

net = SimpleNet(cross_entropy_loss, Adam(),\
    layers=[
        FullConnectedLayer(784, 160),
        ReLULayer(),
        FullConnectedLayer(160, 160),
        DropoutLayer(0.4),
        ReLULayer(),
        FullConnectedLayer(160, 32),
        DropoutLayer(0.4),
        ReLULayer(),
        FullConnectedLayer(32, 10),
        SoftmaxLayer()
    ])

net = load_network('../models/net_160-32_10_98.16.pkl')

with open('mnist_dataset.pkl', 'rb') as f:
    train_dataset, eval_dataset = pickle.load(f)

dataloader = DataLoader(*train_dataset, BATCH_SIZE, EPOCH)

def eval(eval_data, eval_labels):
    a = np.argmax(eval_labels, axis=1)
    b = np.argmax(net.predict(eval_data), axis=1)
    return np.mean(a==b)

for pack in dataloader:
    loss = net.train(*pack)
    
    if dataloader.iter_cnt % 50 == 0:
        net.optimizer.lr = 0.0015 + 0.001 *  np.sin(dataloader.iter_cnt / ( 60000 / BATCH_SIZE ) * 2 * np.pi)
        acc = eval(*eval_dataset)
        print("Epoch %s | Iteration %s | Loss %.4f | Acc %s" %(dataloader.epoch_cnt, dataloader.iter_cnt, loss, acc))

# save_network(net, 'net.pkl')