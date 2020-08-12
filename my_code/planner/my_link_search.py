import numpy as np
import random
import time

###################################################################
class LinkSearch():
    def __init__(self, map_=None):
        self.map = map_

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