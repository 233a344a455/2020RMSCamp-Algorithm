import os
import sys
import time
import numpy as np
from PIL import Image  

sys.path.append("../auto_grader/")
from auto_grader import auto_grader
sys.path.append("../read_picture/")
import read_picture
sys.path.append("../simple_train/")
import simple_net

net = simple_net.load_network('../simple_train/nets/net_128-128_97.74.pkl')

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
num = list(simple_net.one_hot_decode(pred))
# print([{'colour':c, 'num':n} for c, n in zip(colour, num)])

print(np.reshape(num, (8, 8)))
time.sleep(1e8)


