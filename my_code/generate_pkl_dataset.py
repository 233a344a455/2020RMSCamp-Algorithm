import sys
sys.path.append("../read_picture/")
import read_picture

import numpy as np
import pickle

train_data, train_labels = read_picture.read_image_data('../mnist_data/train-images.idx3-ubyte', '../mnist_data/train-labels.idx1-ubyte')
train_data = np.reshape(train_data, (-1, 784)).astype(np.float) / 255
train_labels = np.eye(10)[train_labels]

eval_data, eval_labels = read_picture.read_image_data('../mnist_data/t10k-images.idx3-ubyte', '../mnist_data/t10k-labels.idx1-ubyte')
eval_data = np.reshape(eval_data, (-1, 784)).astype(np.float) / 255
eval_labels = np.eye(10)[eval_labels]

dataset = [[train_data, train_labels], [eval_data, eval_labels]]

with open('mnist_dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)