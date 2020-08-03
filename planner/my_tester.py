import random
import numpy as np

N_KIND = 3 * 10

def generate_map():
    sample_list = random.choices(range(1, N_KIND), k=32) * 2
    return np.array(sample_list, dtype=np.int).reshape(8, 8)



print(generate_map())