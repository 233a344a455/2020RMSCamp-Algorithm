import random
import numpy as np
import matplotlib.pyplot as plt
from simple_net import *

fc1 = FullConnectedLayer(1, 20)
fc2 = FullConnectedLayer(20, 1)
sig = SigmoidLayer(20)
loss_fn = MSELoss()

plt.ion()
for epoch in range(200):
    x = np.linspace(-20, 20)
    y = np.sin(x * 0.2) + np.random.randn(50) * 0.2
    for x_, y_ in random.sample(list(zip(x, y)), 50):

        x = np.linspace(-80, 80, 200)
        y = map(lambda x_: fc2.forward(sig.forward(fc1.forward(x_))),
                [np.array([[x_]]) for x_ in x])
        y = [y_.item() for y_ in y]

        # print(x, y)
        plt.clf()
        plt.plot(x, y, color='red')
        plt.plot(x, np.sin(x*0.2), color='green')
        # plt.show()
        plt.pause(0.001)

        x_ = np.array([[x_]])
        pred_ = fc2.forward(sig.forward(fc1.forward(x_)))
        loss = loss_fn.forward(pred_, y_)
        grad = loss_fn.backward(y_)
        grad = fc2.backward(grad, 5*1e-4, 5*1e-4)
        grad = sig.backward(grad)
        grad = fc1.backward(grad, 5*1e-4, 5*1e-4)