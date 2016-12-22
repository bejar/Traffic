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
from Util.Constants import  dataset_path
from Generate_Dataset import list_days_generator
from keras.utils import np_utils

def load_days(days, z_factor):
    """
    loads and contatenates files from a list of days
    :param days:
    :return:
    """
    ldata = []
    labels = []
    for day in days:
        data = np.load(dataset_path + 'data-D%s-Z%0.2f.npy' % (day, z_factor))
        ldata.append(data)
        labels.extend(np.load(dataset_path + 'labels-D%s-Z%0.2f.npy' % (day, z_factor)))
    data = np.concatenate(ldata)
    return data, labels



def simpleDataGenerator(days, z_factor, batchsize, groups):
    """
    Loops through the day files yielding a batch of examples
    :param days:
    :return:
    """

    while True:
        shuffle(days)
        lgroups = []
        for i in range(0, len(days), groups):
            group = []
            for j in range(groups):
                if (i + j) < len(days):
                    group.append(days[i + j])
            lgroups.append(group)

        for lday in lgroups:
            data, labels = load_days(lday, z_factor)

            limit = (data.shape[0]//batchsize) - 1
            X_train = data.transpose((0,3,1,2))
            y_trainO = [i -1 for i in labels]
            y_train = np_utils.to_categorical(y_trainO, len(np.unique(y_trainO)))

            for i in range(limit):
                yield X_train[i*batchsize:(i+1)*batchsize], y_train[i*batchsize:(i+1)*batchsize]


if __name__ == '__main__':
    ldays = list_days_generator(2016, 11, 1, 12)
    simpleDataGenerator(ldays, 0.25, 100, 5)
