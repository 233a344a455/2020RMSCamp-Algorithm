import random
import numpy as np
import matplotlib.pyplot as plt
from simple_net_mutiple import *

net = SimpleNet(MSELoss, layers=[
    FullConnectedLayer(1, 20, 0.3*1e-4),
    LeakyReLULayer(0.05),
    FullConnectedLayer(20, 5, 0.3*1e-4),
    LeakyReLULayer(0.03),
    FullConnectedLayer(5, 1, 1e-4)
])

plt.ion()
z = 0
x = np.linspace(-20, 20)
for epoch in range(20000):
    y = np.sin(x*0.3)+3 + np.random.randn(50)

    loss = net.train(x[:, np.newaxis], y[:, np.newaxis])

    z += 1
    if z % 500 == 0:
        print(loss)

        plt.clf()
        plt.plot(x, net.predict(x[:, np.newaxis]), color='red')
        plt.plot(x, sin(x*0.3)+3, color='green')

        plt.pause(0.01)




