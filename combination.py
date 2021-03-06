#!/usr/bin/env python

import itertools
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

import logging

logfile = "{}.log".format(datetime.now().strftime("%Y%m%d_%H%M%S"))

logging.basicConfig(level = logging.INFO, filename = logfile)

boxes = int(input("total number of boxes: "))

box = np.zeros(boxes+1)

for num_box in range(1, boxes+1):
    #balls = num_box
    print("num of boxes: ", num_box)
    logging.info("num of boxes: {}".format(num_box))

    total_rule = 0
    total = []

    for balls in range(0, num_box + 1):
        rng = []
        rng = list(range(balls+1)) * num_box
        #p = set(i for i in itertools.permutations(rng, num_box) if sum(i) == balls)
        p = []

        for i in itertools.permutations(rng, num_box):
            if sum(i) == balls:
                total.append(i)
                flag = True
                #print(i, len(i))
                for t in range(0, len(i)-1):
                    if (i[t]>t+1):
                        flag = False
                if (flag):
                    p.append(i)
                print(".", end="")
        rule_p = set(p)

        #print("    total: ", len(p))
        '''
        for i in p:
            flag = True
            #print(i, len(i))
            for t in range(0, len(i)-1):
                if (i[t]>t+1):
                    flag = False
            if (flag):
                rule_p.append(i)
            print(".", end="")
        '''
        #print(rule_p)
        print("    ruled: ", len(rule_p), "\n")
        #box[num_box] = len(rule_p)
        #total_num += len(rule_p)
        logging.info("{} balls has {} ways".format(balls, len(rule_p)))
        logging.info("all combinations: {}".format(rule_p))
        total_rule = total_rule + len(rule_p)
        
    
    logging.info("total ruled: {}".format(total_rule))
    total_ways = len(set(total))
    logging.info("total ways: {}, and ratio is {}\n".format(total_ways, total_rule/total_ways))

#plt.figure(1)
#k = np.arange(0, boxes+1, 1)

#plt.plot(k, box)
#plt.show()
