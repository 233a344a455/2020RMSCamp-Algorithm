import os
import sys
import time
import numpy as np
from PIL import Image

sys.path.append("../auto_grader/")
from auto_grader import auto_grader

sys.path.append("./net/")
import simple_net

net1 = simple_net.load_network('./models/net_160-32_10_98.16.pkl')
net2 = simple_net.load_network('./models/net_128-128_98.1.pkl')

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

img_list = np.array(img_list, dtype=np.float) / 255

raw_pred = net1.predict(img_list) * 1.3 + net2.predict(img_list)

pred = np.reshape(simple_net.one_hot_decode(raw_pred), (8, 8))
colour = np.reshape(colour, (8, 8))

main_map = (colour + 1) * (pred + 1)

unique, counts = np.unique(main_map, return_counts=True)
print(counts)
print(np.sum(counts % 2 != 0))

print(pred)
print(colour)
print(main_map)

time.sleep(1e8)