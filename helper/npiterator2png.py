# npiterator2png.py
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

imgs = data['input_raw_data']
for i in range(23280):
    img = imgs[i].reshape(100,100)
    misc.imsave('data/img'+str(i)+'.png', img)
