from planner import my_link_search
from time import sleep
import numpy as np
import itertools
import heapq
import random
from functools import total_ordering

import sys
sys.path.append('../auto_grader')
import auto_grader

def packbits(bool_array):
    bits = 0
    for idx, bit in enumerate(bool_array):
        bits += bit << idx
    return bits

def unpackbits(bits):
    bool_array = []
    for idx in range(64):
        bool_array.append((bits >> idx) & 1)
    return np.array(bool_array, dtype=np.bool)

class LinkSearch():
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

    def search_link(self, map_, x1, y1, x2, y2):
        self.map = map_
        if x1 == x2 and y1 == y2:
            return None
        if self.direct(x1, y1, x2, y2):
            # print('1 line.')
            return 1
        elif self.one_corner(x1, y1, x2, y2):
            # print('2 lines.')
            return 2
        elif self.two_corner(x1, y1, x2, y2):
            # print('3 lines.')
            return 3
        elif self.three_corner(x1, y1, x2, y2):
            # print('4 lines.')
            return 4
        else:
            # print('more than 4 lines.')
            return None
ls = LinkSearch()


@total_ordering
class Node():
    def __init__(self, parent_node, link, unvisited_pos, cost):
        self.parent_node = parent_node
        self.cost = cost
        self.link = link
        self.unvisited_pos = unvisited_pos
        
    def __str__(self):
        return ("(%s, %s) -> (%s, %s)" %(self.link[0], self.link[1], self.link[2], self.link[3]))

    def __eq__(self, other):
        return self.cost == other.cost
    
    def __lt__(self, other):
        return self.cost < other.cost

class Astar():
    def __init__(self, map_orig):
        self.map_orig = map_orig
        self.visited_nodes = []
        self.priority_heap = []
    
    def get_map(self, node):
        unvisited_pos = np.reshape(unpackbits(node.unvisited_pos), (8, 8))
        unvisited_pos = np.pad(unvisited_pos, ((1,1),(1,1)), 'constant', constant_values=(0,0))
        return self.map_orig * unvisited_pos
        
    def find_available_nodes(self, node):
        map_ = self.get_map(node)
        for i in range(1, 31): # [0, 9] + [0, 2] * 10 + 1 -> [1, 30]
            pts = np.argwhere(map_ == i)
            for link in itertools.combinations(pts, 2):
                link = np.concatenate(link)
                cost = ls.search_link(map_, *link)
                if cost is not None:
                    unvisited_pos = self.remove_unvisited_pos(node.unvisited_pos, *link)
                    if unvisited_pos not in self.visited_nodes:
                        n = Node(node, link, unvisited_pos, node.cost + cost - 2)
                        heapq.heappush(self.priority_heap, n)
                        self.visited_nodes.append(unvisited_pos)
    
    def remove_unvisited_pos(self, last_unvisited_pos, x1, y1, x2, y2):
            unvisited_pos = unpackbits(last_unvisited_pos)
            # print(np.sum(unvisited_pos))
            if not unvisited_pos.any():
                print("End!!!")
                exit()
            unvisited_pos[(x1 - 1) * 8 + y1 - 1] = unvisited_pos[(x2 - 1) * 8 + y2 - 1] = 0
            return packbits(unvisited_pos)

    def main(self):
        init_node = Node(None, None, packbits(np.ones(64, dtype=np.bool)), 0)
        heapq.heappush(self.priority_heap, init_node)
        while len(self.priority_heap):
            n = heapq.heappop(self.priority_heap)
            self.find_available_nodes(n)


def generate_map():
    sample_list = random.choices(range(1, 30), k=32) * 2
    return np.array(sample_list, dtype=np.int).reshape(8, 8)

map_orig = np.pad(generate_map(), ((1,1),(1,1)), 'constant', constant_values=(0,0))
a_star = Astar(map_orig)
a_star.main()


# ag = auto_grader.auto_grader()

# raw_map = ag.get_map()
# colour_map = np.reshape(np.array(raw_map)[:, 0], (8, 8))
# num_map = np.reshape(np.array(raw_map)[:, 1], (8, 8))
# map_ = num_map + colour_map * 10 + 1
# map_ = np.pad(map_, ((1,1),(1,1)), 'constant', constant_values=(0,0))