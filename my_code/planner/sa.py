
import random
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt


class LinkSearch():
    def link(self, x1, y1, x2, y2):
        self.map[x1, y1] = self.map[x2, y2] = 0

    def direct(self, x1, y1, x2, y2):
        if x1 == x2 and y1 == y2:
            return False
        elif x1 == x2:    # 平行y轴
            return not self.map[x1, y1:y2:(2*(y1 < y2)-1)][1:].any()
        elif y1 == y2:    # 平行x轴
            return not self.map[x1:x2:(2*(x1 < x2)-1), y1][1:].any()
        return False

    def is_empty(self, x, y):
        return not self.map[x, y]

    def get_boarder(self, x, y, direction):
        if direction=='x':
            a = b = x
            while a>0 and self.is_empty(a-1, y):
                a -= 1
            while b<9 and self.is_empty(b+1, y):
                b += 1
        elif direction=='y':
            a = b = y
            while a>0 and self.is_empty(x, a-1):
                a -= 1
            while b<9 and self.is_empty(x, b+1):
                b += 1
        return a, b + 1

    def one_corner(self, x1, y1, x2, y2):
        p1 = self.direct(x1, y2, x1, y1) and self.direct(x1, y2, x2, y2) and not self.map[x1, y2] # x1, y2
        p2 = self.direct(x2, y1, x1, y1) and self.direct(x2, y1, x2, y2) and not self.map[x2, y1] # x2, y1
        return p1 or p2
        
    def two_corner(self, x1, y1, x2, y2):
        a1, b1 = self.get_boarder(x1, y1, 'y')
        a2, b2 = self.get_boarder(x2, y2, 'y')
        for y in range(max(a1, a2), min(b1, b2)):
            if self.direct(x1, y, x2, y):
                return True
        
        a1, b1 = self.get_boarder(x1, y1, 'x')
        a2, b2 = self.get_boarder(x2, y2, 'x')
        for x in range(max(a1, a2), min(b1, b2)):
            if self.direct(x, y1, x, y2):
                return True

        return False

    def three_corner(self, x1, y1, x2, y2):
        ax1, bx1 = self.get_boarder(x1, y1, 'x')
        ay2, by2 = self.get_boarder(x2, y2, 'y')
        ax2, bx2 = self.get_boarder(x2, y2, 'x')
        ay1, by1 = self.get_boarder(x1, y1, 'y')
        
        for x in range(ax1, bx1):
            for y in range(ay2, by2):
                if self.is_empty(x, y) and self.direct(x2, y, x, y) and self.direct(x, y, x, y1):
                    return True

        for x in range(ax2, bx2):
            for y in range(ay1, by1):
                if self.is_empty(x, y) and self.direct(x1, y, x, y) and self.direct(x, y, x, y2):
                    return True
        
        return False

    def search_link(self, x1, y1, x2, y2):
        if x1 == x2 and y1 == y2:
            return None
        if self.direct(x1, y1, x2, y2):
            return 50
        elif self.one_corner(x1, y1, x2, y2):
            return 20
        elif self.two_corner(x1, y1, x2, y2):
            return 10
        elif self.three_corner(x1, y1, x2, y2):
            return 0
        else:
            return None
    
    def is_all_empty(self):
        return not self.map.any()

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
        print(score)

        T *= q

    return best_score, best_path

def generate_map():
    sample_list = random.choices(range(1, 30), k=32) * 2
    return np.array(sample_list, dtype=np.int).reshape(8, 8)

if __name__ == "__main__":
    map_orig = np.pad(generate_map(), ((1,1),(1,1)), 'constant', constant_values=(0,0))
    simulated_annealing(map_orig)
