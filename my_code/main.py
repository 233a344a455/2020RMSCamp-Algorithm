import os
import sys
import time
import numpy as np
from PIL import Image

sys.path.append("../auto_grader/")
from auto_grader import auto_grader

sys.path.append("./net/")
import simple_net

net = simple_net.load_network('./models/net_128-128_98.1.pkl')

ag = auto_grader()

colour = []
img_list = []
for idx in range(64):
    img = np.array(Image.open("../auto_grader/image/%s.png" %(idx)))
    for color_idx in range(3):
        img_arr = img[:, :, color_idx]
        if (img_arr != 0).any():
            colour.append(color_idx)
            img_list.append(img_arr.flatten())
            break

pred = net.predict(np.array(img_list, dtype=np.float) / 255)

print(np.reshape(simple_net.one_hot_decode(pred), (8, 8)))
print(np.reshape(colour, (8, 8)))

time.sleep(1e8)