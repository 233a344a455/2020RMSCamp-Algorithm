import random
import numpy as np
# import itertools

# def SA():
#     # init path
#     path = list(range(1, 36))

#     # notes
#     # q = 0.98, tb = 6000, te = T_begin * (q ** 300), m_l = 1000
#     # q = 0.99, tb = 3000, te = T_begin * (q ** 400), m_l = 1000

#     # Cool down speed
#     q = 0.99

#     T_begin = 3000.0
#     T_end = T_begin * (q ** 400)
#     T = T_begin

#     # Iteration per Temp
#     mapkob_len = 2000

#     # global min_path_len
#     # min_path_len = []

#     best_path_l = 1e8

#     while T > T_end:
#         for _ in range(mapkob_len):

#             new_path = create_new_path(path)
#             path_l = path_len(path)
#             new_path_l = path_len(new_path)
#             df = new_path_l - path_l

#             if df < 0:
#                 path = new_path[:]
#                 if path_l < best_path_l:
#                     best_path = path[:]
#                     best_path_l = path_l

#             elif math.exp(-df/T) > random.random():
#                 path = new_path[:]

#         T *= q

#     best_path.insert(0, 0)
#     return best_path, best_path_l / 1000.0




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
            # print('1 line.') 50
            return 50
        elif self.one_corner(x1, y1, x2, y2):
            # print('2 lines.') 20
            return 20
        elif self.two_corner(x1, y1, x2, y2):
            # print('3 lines.') 10
            return 10
        elif self.three_corner(x1, y1, x2, y2):
            # print('4 lines.') 0
            return 0
        else:
            # print('more than 4 lines.')
            return None
    
    def is_all_empty(self):
        return not self.map.any()

ls = LinkSearch()

def generate_map():
    sample_list = random.choices(range(1, 30), k=32) * 2
    return np.array(sample_list, dtype=np.int).reshape(8, 8)

map_orig = np.pad(generate_map(), ((1,1),(1,1)), 'constant', constant_values=(0,0))

rand_list = [(x, y) for x in range(1, 10) for y in range(1, 10)]
def create_new_path():
    global rand_list
    ls.map = map_orig.copy()
    rand_range_len = random.randrange(0, 8)
    a = random.randrange(0, 64 - rand_range_len)
    b = a + rand_range_len
    rand_list[a:b] = random.sample(rand_list[a:b], k=rand_range_len)

    score = 0
    path = []

    for x, y in rand_list:
        num = ls.map[x, y]
        if num == 0:
            continue
        
        for p2 in np.argwhere(ls.map == num):
            s = ls.search_link(x, y, *p2)
            if s is not None:
                ls.link(x, y, *p2)
                path.append([x, y, p2[0], p2[1]])
                score += s
        
    if ls.is_all_empty:
        return path, score
    else:
        return None

for _ in range(20):
    # print(create_new_path())
    pass