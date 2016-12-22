"""
.. module:: DataGenerators

DataGenerators
*************

:Description: DataGenerators

    

:Authors: bejar
    

:Version: 

:Created on: 21/12/2016 8:13 

"""

__author__ = 'bejar'

from numpy.random import shuffle
import numpy as np
from Util.Constants import data_path, dataset_path

def simpleDataGenerator(days, z_factor, batchsize):
    """
    Loops through the day files yielding a batch of examples
    :param days:
    :return:
    """

    while True:
        shuffle(days)
        for day in days:
            data = np.load(dataset_path + 'data-D%s-Z%0.2f-C%d.npy' % (day, z_factor))
            labels = np.load(dataset_path + 'data-D%s-Z%0.2f-C%d.npy' % (day, z_factor))
            limit = (data.shape[0]//batchsize) - 2

            for i in range(limit):
                yield data[i, (i+1)*batchsize], labels[i, (i+1)*batchsize]

