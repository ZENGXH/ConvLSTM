# clips2powertxt.py

import numpy
from scipy import misc
import theano
import theano.tensor as TT

import sparnn
import sparnn.utils
from sparnn.utils import *

from sparnn.iterators import NumpyIterator

import os
import random
import numpy
import logging

## for test data
if(mode == "test"):
	iterator_param = {'path': '../../SPARNN/data/hko-example/hko-test.npz',
	                  'minibatch_size': 8,
	                  'use_input_mask': False,
	                  'input_data_type': 'float32',
	                  'is_output_sequence': True,
	                  'name': 'hko-test-iterator'}
	test_iterator = NumpyIterator(iterator_param)
	test_iterator.begin(do_shuffle=False)
	test_iterator.print_stat()
	data = test_iterator.data
elif(mode == "train"):

elif(mode == "valid"):


imgs = data['input_raw_data']
index = data['clips']
startingP_input = [i[0] for i in index[0]]
startingP_output = [i[0] for i in index[1]]
print(startingP_input[0:3])
print(startingP_output[0:3])

print("number of starting points ", len(startingP))
print("open " + mode + "seq.txt")

f = open( mode + "seq.txt",'w')

for i in range(len(startingP_input)):
    # ind = startingP
    for frames in range(5):
        f.write('img'+str(startingP_input[i]+frames)+'.png')
        f.write('\n')
    for frames in range(15):
        f.write('img'+str(startingP_output[i]+frames)+'.png')
        f.write('\n')
    # f.write('\n')
    # if(i == 4):
    #    break
f.close()
print("done")

