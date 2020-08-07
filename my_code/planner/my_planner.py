import numpy as np
import random
import time

N_KIND = 3 * 10

def generate_map():
    sample_list = random.choices(range(1, N_KIND), k=32) * 2
    return np.array(sample_list, dtype=np.int).reshape(8, 8)

main_map = generate_map()
main_map = np.pad(main_map,((1,1),(1,1)),'constant',constant_values = (0,0)) 
print(main_map)

###################################################################

def link(x1, y1, x2, y2):
        print("Linked (%s, %s) to (%s, %s)." % (x1, y1, x2, y2))
        main_map[x1, y1] = 0
        main_map[x2, y2] = 0

def direct(x1, y1, x2, y2):
    if x1 == x2 and y1 != y2:    # 平行y轴
        return not main_map[x1, y1:y2:(2*(y1 < y2)-1)][1:].any()
    if y1 == y2 and x1 != x2:    # 平行x轴
        return not main_map[x1:x2:(2*(x1 < x2)-1), y1][1:].any()
    return False

def one_corner(x1, y1, x2, y2):
    r1 = direct(x1, y2, x1, y1) and direct(x1, y2, x2, y2) and not main_map[x1, y2] # x1, y2
    r2 = direct(x2, y1, x1, y1) and direct(x2, y1, x2, y2) and not main_map[x2, y1] # x2, y1
    return r1 or r2
    
def two_corner(x1, y1, x2, y2):
    for x in range(10):  # (x, y1) (x, y2)
        if not(main_map[x, y1] or main_map[x, y2])\
        and direct(x1, y1, x, y1) and direct(x, y1, x, y2) and direct(x, y2, x2, y2):
            return True

    for y in range(10):  # (x1, y) (x2, y)
        if not(main_map[x1, y] or main_map[x2, y])\
        and direct(x1, y1, x1, y) and direct(x1, y, x2, y) and direct(x2, y, x2, y2):
            return True
    
    return False

def three_corner(x1, y1, x2, y2):
    if x1 == x2 and y1 == y2:
        return False
    for x in range(10):
        for y in range(10):
            p1 = not (main_map[x1, y] or main_map[x, y] or main_map[x, y2]) and\
                direct(x1, y1, x1, y) and direct(x1, y, x, y) and direct(x, y, x, y2) and direct(x, y2, x2, y2)
            p2 = not (main_map[x, y1] or main_map[x, y] or main_map[x2, y]) and\
                direct(x1, y1, x, y1) and direct(x, y1, x, y) and direct(x, y, x2, y) and direct(x2, y, x2, y2)
            if p1 or p2:
                return True
    return False

def get_score(x1, y1, x2, y2):
    if direct(x1, y1, x2, y2):
        print('1 line.')
        return 50
    elif one_corner(x1, y1, x2, y2):
        print('2 lines.')
        return 10
    elif two_corner(x1, y1, x2, y2):
        print('3 lines.')
        return 20
    elif three_corner(x1, y1, x2, y2):
        print('4 lines.')
        return 0
    else:
        print('more than 4 lines.')
        return None

###################################################################

# print(main_map)

# 消除相邻的
for i in range(1, 10):
    for j in range(1, 10):
        cur_num = main_map[i, j]

        if cur_num == 0:
            continue

        try:
            if main_map[i+1, j] == cur_num:
                link(i, j, i+1, j)
        except IndexError:
            pass
        try:
            if main_map[i-1, j] == cur_num:
                link(i, j, i-1, j)
        except IndexError:
            pass

        try:
            if main_map[i, j+1] == cur_num:
                link(i, j, i, j+1)
        except IndexError:
            pass

        try:
            if main_map[i, j-1] == cur_num:
                link(i, j, i, j-1)
        except IndexError:
            pass

print(main_map)

score = 0
while main_map.any():
    for x in range(1, 10):
        for y in range(1, 10):
            num = main_map[x, y]
            if num == 0:
                continue
            
            max, maxidx = -1, None
            for p2 in np.argwhere(main_map == num):
                s = get_score(x, y, *p2)
                if s is not None:
                    if s > max:
                        max, maxidx = s, p2

            if maxidx is not None:
                score += max
                link(x, y, *maxidx)
                print(main_map)
                continue

print(score)