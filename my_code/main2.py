import os
import sys
import time
import numpy as np
from PIL import Image

sys.path.append("../auto_grader/")
from auto_grader import auto_grader

sys.path.append("./net/")
import simple_net

sys.path.append("./planner/")
import simple_net

net = simple_net.load_network('./models/net.pkl')

ag = auto_grader(False)

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


unique, counts = np.unique(main_map, return_counts=True)
print(dict(zip(unique, counts)))

wrong_value = unique[np.argwhere(counts % 2 != 0)][:, 0]
if len(wrong_value) > 2:
    raise Exception("Too many errors!")
elif len(wrong_value) == 2:

    min_pro, min_pro_idx, min_pro_value = 1, None, None
    for v in wrong_value:
        wrong_idx = np.argwhere(main_map == v)
        for i in wrong_idx:
            pro = np.max(raw_pred[i[0] * 8 + i[1]])
            if pro < min_pro:
                min_pro, min_pro_idx, min_pro_value = pro, i, v
            print("%s probability = %s" %(i,  pro))
    
    correct_value = wrong_value[wrong_value != min_pro_value][0]
    print("Waring: idx %s: Corrected %s to %s." %(min_pro_idx, min_pro_value, correct_value))
    main_map[min_pro_idx[0], min_pro_idx[1]] = correct_value

print(main_map)

sco, path = simulated_annealing(map_, )

for l in path:
    ag.link(*l)
# true_num_map  = np.reshape(np.array(ag.get_map())[:, 1], (8, 8))
# wrong_idx = np.argwhere(num_map != true_num_map)
# for idx in wrong_idx:
#     print("idx: %s, pred %s, target %s" %(idx, num_map[idx[0], idx[1]], true_num_map[idx[0], idx[1]]))

# time.sleep(1e8)