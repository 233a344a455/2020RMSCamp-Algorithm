import os
import sys
import time
import numpy as np
from PIL import Image  

# sys.path.append("../auto_grader/")
# from auto_grader import auto_grader
sys.path.append("../read_picture/")
import read_picture
sys.path.append("../simple_train/")
import simple_train

    
# 读入图片
train_image, train_label = read_picture.read_image_data('../mnist_data/train-images.idx3-ubyte', '../mnist_data/train-labels.idx1-ubyte')
train_image_vector = np.reshape(train_image, (60000, 784))
trainer = simple_train.simple_train_one_num(train_image_vector[0:5000], train_label[0:5000], 10, 0.1, 2.55)
trainer.train_learn()

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

num = list(trainer.predict(img_list))
print([{'colour':c, 'num':n} for c, n in zip(colour, num)])