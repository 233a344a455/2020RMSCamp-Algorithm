import random
import numpy as np
import matplotlib.pyplot as plt
from simple_net import *

net = SimpleNet(MSELoss, layers=[
    FullConnectedLayer(1, 5, 1e-4),
    LeakyReLULayer(0.05),
    FullConnectedLayer(5, 5, 1e-4),
    LeakyReLULayer(0.03),
    FullConnectedLayer(5, 1, 1e-4)
])

plt.ion()
z = 0
for epoch in range(200):
    x = np.linspace(-20, 20)
    y = x ** 3 * 0.01 + 5 + np.random.randn(50) * 2
    for x_, y_ in random.sample(list(zip(x, y)), 50):

        x_, y_ = np.array([[x_]]), np.array([[y_]])
        loss = net.train(x_, y_)
        # pred_ = fc2.forward(relu.forward(fc1.forward(x_)))
        # loss = loss_fn.forward(pred_, y_)
        # grad = loss_fn.backward(y_)
        # grad = fc2.backward(grad, 1*1e-4)
        # grad = relu.backward(grad)
        # grad = fc1.backward(grad, 1*1e-4)

        z += 1
        if z % 10 == 0:
            print(loss)
            x = np.linspace(-20, 20, 200)
            y = map(lambda x_: net.predict(x_),
                    [np.array([[x_]]) for x_ in x])
            y = [y_.item() for y_ in y]

            # print(x, y)
            plt.clf()
            plt.plot(x, y, color='red')
            plt.plot(x, x**3 * 0.01 + 5, color='green')
            # plt.show()
            plt.pause(0.01)




