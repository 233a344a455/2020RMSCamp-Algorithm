
import random
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt

from my_link_search import LinkSearch
ls = LinkSearch()

def create_new_path(map_orig, rand_list):
    ls.map = map_orig.copy()

    # Create new rand list
    for _ in range(5):
        a = random.randrange(0, 64)
        b = random.randrange(0, 64)
        rand_list[a], rand_list[b] = rand_list[b], rand_list[a]

    score = 0
    path = []

    for x, y in rand_list:
        num = ls.map[x, y]
        if num == 0:
            continue

        sl, pl = [], []
        for p in np.argwhere(ls.map == num):
            s = ls.search_link(x, y, *p)
            if s is not None:
                sl.append(s)
                pl.append(p)

        if len(sl):
            idx = sl.index(max(sl))
            path.append([x-1, y-1, pl[idx][0]-1, pl[idx][1]-1])
            ls.link(x, y, pl[idx][0], pl[idx][1])
            score += sl[idx]
            # 第4次，第8次，第16次，第28，29，30，31，32次配对
            if len(path) in (4, 8, 16, 28, 29, 30, 31, 32) and sl[idx] == 50:
                score += 50
    
    if ls.is_all_empty():
        return score, path, rand_list
    else:
        return None


score_list = []
def simulated_annealing(map_orig, q = 0.98, T_begin = 50, T_end = 5, mapkob_len = 50):

    map_orig = np.pad(map_orig, ((1,1),(1,1)),'constant',constant_values = (0,0)) 

    best_path = None
    best_score = 0
    T = T_begin
    rand_list = [(x, y) for x in range(1, 10) for y in range(1, 10)]
    score = 0

    last_ret = None
    ret = None

    while T > T_end:
        for _ in range(mapkob_len):
            
            while ret is None or ret == last_ret:
                ret = create_new_path(map_orig, rand_list.copy())
            last_ret = ret
            new_score, new_path, new_rand_list = ret

            df = score - new_score

            if df < 0:
                path = new_path
                score = new_score
                rand_list = new_rand_list

                if score > best_score:
                    best_score, best_path = score, path

            elif math.exp(-df/T) > random.random():
                path = new_path
                score = new_score
                rand_list = new_rand_list
            
            score_list.append(score)
        
        plt.clf()
        plt.plot(score_list)
        plt.pause(0.05)
        # print(score)

        T *= q

    return best_score, best_path

def generate_map():
    sample_list = random.choices(range(1, 30), k=32) * 2
    return np.array(sample_list, dtype=np.int).reshape(8, 8)

if __name__ == "__main__":
    map_orig = generate_map()
    simulated_annealing(map_orig)
