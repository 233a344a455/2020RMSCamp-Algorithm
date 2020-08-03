from my_tester import *

import numpy as np
import random
import time

main_map = generate_map()

def link(x1, y1, x2, y2):
    print("Linked (%s, %s) to (%s, %s)." % (x1, y1, x2, y2))
    main_map[x1, y1] = 0
    main_map[x2, y2] = 0

# print(main_map)

# 消除相邻的
for i in range(8):
    for j in range(8):
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