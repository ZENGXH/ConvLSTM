import re
import scipy
import pylab
from matplotlib import pyplot as plt
import numpy as np
import sys
import math
import glob
from sparnn.helpers import visualization
# print "argument2 is " + sys.argv[2]
# fil = open('hko-record/HKO-prediction-'+ sys.argv[1] + '.log','r')
# fil
# print fil

path = "of_record/"
#fileList = glob.glob(path + "*out*")
#for fil in fileList:
    # print fil
    # tx = fil.read()
#    visualization.visualize_loss(fil)

fileList = glob.glob(path + "out_of11")
for target_file in fileList:
#    print(target_file)
    loss_file = open(target_file, 'r')
    lines = loss_file.readlines()
    loss_file.close()

    iter_list = []
    loss_list = []

    iter_num = 0
    flag = 1
    accum_loss = 0
    accum_loss_list = []
    accum_loss_iter_list = []
    average_interval = 500

    for ind, line in enumerate(lines):
#        if "err" in line:
#            print(line)
        if "valid" in line:
	    print(line)
            continue
        if "err" not in line:
            continue
#        print(line)

        loss_num = line.split('err:\t')[1]

#        print(loss_num)
        #loss_num = 0
        iter_num += 1
        loss_list.append(float(loss_num))
        iter_list.append(int(iter_num))

        accum_loss += float(loss_num)
        if 0 == (int(iter_num) % average_interval):
            accum_loss_iter_list.append(int(iter_num))
            accum_loss_list.append(float(accum_loss))
            accum_loss = 0
    # plt.plot(iter_list[:], loss_list[:])
    plt.xlabel('iteration number')
    plt.ylabel('training loss')
    # plt.clf()

    plt.plot(accum_loss_iter_list[:], accum_loss_list[:])
    plt.title(target_file)
    plt.grid()
    plt.savefig('vis_training_loss_'  + '.png')
    plt.show()


