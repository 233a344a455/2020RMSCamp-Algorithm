import numpy as np
import random
import time

###################################################################

class MyLinkSearch():
    def __init__(self, map_, auto_grader=None):
        self.auto_grader = auto_grader
        self.pretreat(map_)

    def link(self, x1, y1, x2, y2):
            print("Linked (%s, %s) to (%s, %s)." % (x1, y1, x2, y2))
            if self.auto_grader:
                ret = self.auto_grader.link(x1-1, y1-1, x2-1, y2-1)
                self.map[x1, y1] = self.map[x2, y2] = 0
                if ret > 0:
                    return ret
                else:
                    raise Exception("ret == %s" %ret)

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
            print('1 line.')
            return 50
        elif self.one_corner(x1, y1, x2, y2):
            print('2 lines.')
            return 20
        elif self.two_corner(x1, y1, x2, y2):
            print('3 lines.')
            return 10
        elif self.three_corner(x1, y1, x2, y2):
            print('4 lines.')
            return 0
        else:
            print('more than 4 lines.')
            return None
    
    def pretreat(self, map_):
        self.map = np.pad(map_,((1,1),(1,1)),'constant',constant_values = (0,0)) 

###################################################################

def generate_map():
    sample_list = random.choices(range(1, 30), k=32) * 2
    return np.array(sample_list, dtype=np.int).reshape(8, 8)

if __name__ == "__main__":

    main_map = generate_map()
    ls = MyLinkSearch(main_map)

    score = 0
    while main_map.any():
        for x in range(1, 10):
            for y in range(1, 10):
                num = main_map[x, y]
                if num == 0:
                    continue
                
                max_sco, maxidx = -1, None
                for p2 in np.argwhere(main_map == num):
                    s = search_link(x, y, *p2)
                    if s is not None:
                        if s > max_sco:
                            max_sco, maxidx = s, p2

                if maxidx is not None:
                    score += max_sco
                    link(x, y, *maxidx)
                    print(main_map)
                    continue

    print(score)