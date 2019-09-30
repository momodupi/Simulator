#!/usr/bin/env python

import itertools
import numpy as np
import matplotlib.pyplot as plt


boxes = 10

box = np.zeros(boxes+1)

for num_box in range(1, boxes+1):
    #balls = num_box
    print("num of boxes: ", num_box)

    for balls in range(0, num_box + 1):
        rng = []
        rng = list(range(balls+1)) * num_box
        p = set(i for i in itertools.permutations(rng, num_box) if sum(i) == balls)

        #print("    total: ", len(p))
        rule_p = []

        for i in p:
            flag = True
            #print(i, len(i))
            for t in range(0, len(i)-1):
                if (i[t]>t+1):
                    flag = False
            if (flag):
                rule_p.append(i)

        print(rule_p)
        print("    ruled: ", len(rule_p), "\n")
        box[num_box] = len(rule_p)

        
plt.figure(1)
k = np.arange(0, boxes+1, 1)

#plt.plot(k, box)
#plt.show()