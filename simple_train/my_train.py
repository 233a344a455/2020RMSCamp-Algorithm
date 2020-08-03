import random
import numpy as np
import matplotlib.pyplot as plt
from simple_net import *

net = SimpleNet(MSE_loss, BGD(learning_rate=5 * 1e-3),\
                                layers=[
                                    FullConnectedLayer(1, 20),
                                    SigmoidLayer(),
                                    FullConnectedLayer(20, 5),
                                    SigmoidLayer(),
                                    FullConnectedLayer(5, 1)
                                ])

plt.ion()
z = 0
for epoch in range(2000000):
    x = np.linspace(-20, 20, 50)
    y = np.sin(x*0.1)+3 + np.random.randn(50)

    loss = net.train(x[:, np.newaxis], y[:, np.newaxis])

    z += 1
    if z % 500 == 0:
        print(loss)

        plt.clf()
        x = np.linspace(-20, 20, 100)
        plt.plot(x, net.predict(x[:, np.newaxis]), color='red')
        plt.plot(x, np.sin(x*0.1)+3, color='green')

        plt.pause(0.01)




