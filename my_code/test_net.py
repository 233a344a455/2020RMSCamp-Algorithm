import os
import sys
import time
import numpy as np
from PIL import Image

sys.path.append("../auto_grader/")
from auto_grader import auto_grader

sys.path.append("./net/")
import simple_net

net = simple_net.load_network('./models/net.pkl')

ag = auto_grader()

colour_map = []
img_list = []
for idx in range(64):
    img = np.array(Image.open("../auto_grader/image/%s.png" %(idx)))
    for color_idx in range(3):
        img_arr = img[:, :, color_idx]
        if (img_arr != 0).any():
            colour_map.append(color_idx)
            img_list.append(img_arr.flatten())
            break

img_list = np.array(img_list, dtype=np.float) / 255

raw_pred = net.predict(img_list)

num_map = np.reshape(simple_net.one_hot_decode(raw_pred), (8, 8))
colour_map = np.reshape(colour_map, (8, 8))

main_map = num_map + 10 * colour_map + 1

true_num_map  = np.reshape(np.array(ag.get_map())[:, 1], (8, 8))

print(num_map)
wrong_idx = np.argwhere(num_map != true_num_map)
if len(wrong_idx):
    for idx in wrong_idx:
        print("idx: %s, pred %s, target %s" %(idx, num_map[idx[0], idx[1]], true_num_map[idx[0], idx[1]]))
else:
    print("All correct.")

time.sleep(1e5)