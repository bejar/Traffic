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
from Util.Generate_Dataset import list_days_generator
from keras.utils import np_utils

def load_days(days, z_factor, reb=False):
    """
    loads and contatenates files from a list of days
    :param days:
    :return:
    """
    ldata = []
    labels = []
    fnamed = 'data'
    fnamel = 'labels'
    if reb:
        fnamed = 'r' + fnamed
        fnamel = 'r' + fnamel
    for day in days:
        data = np.load(dataset_path + fnamed + '-D%s-Z%0.2f.npy' % (day, z_factor))
        ldata.append(data)
        labels.extend(np.load(dataset_path + fnamel + '-D%s-Z%0.2f.npy' % (day, z_factor)))
    data = np.concatenate(ldata)
    return data, labels


def simpleDataGenerator(days, z_factor, nclasses, batchsize, groups, imgord='th'):
    """
    Loops through the day files yielding a batch of examples
    Files are loaded in groups and batches are randomized
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
            if imgord == 'th':
                X_train = data.transpose((0,3,1,2))
            y_trainO = [i -1 for i in labels]
            y_train = np_utils.to_categorical(y_trainO, nclasses)
            perm = range(X_train.shape[0])
            shuffle(perm)
            lperm = []
            for i in range(0, len(perm), batchsize):
                gperm = []
                for j in range(batchsize):
                    if (i + j) < len(perm):
                        gperm.append(perm[i + j])
                lperm.append(gperm)

            # for i in range(limit):
            #     yield X_train[i*batchsize:(i+1)*batchsize], y_train[i*batchsize:(i+1)*batchsize]

            for i in range(limit):
                yield X_train[lperm[i]], y_train[lperm[i]]


def dayGenerator(day, z_factor, nclasses, batchsize, reb=False, imgord='th'):
    """
    Load the data for a day and return a random permutation for
    generating the random batches

    :param day:
    :param z_factor:
    :param nclasses:
    :param batchsize:
    :return:
    """
    data, labels = load_days([day], z_factor, reb=reb)

    if imgord == 'th':
        X_train = data.transpose((0,3,1,2))
    # For now the generated datasets not rebalanced has labels [1..maxclass]
    if not reb:
        y_trainO = [i -1 for i in labels]
    else:
        y_trainO = labels
    y_train = np_utils.to_categorical(y_trainO, nclasses)
    perm = range(X_train.shape[0])
    shuffle(perm)
    lperm = []
    for i in range(0, len(perm), batchsize):
        gperm = []
        for j in range(batchsize):
            if (i + j) < len(perm):
                gperm.append(perm[i + j])
        lperm.append(gperm)

    return X_train, y_train, lperm

if __name__ == '__main__':
    ldays = list_days_generator(2016, 11, 12, 12)
    gen = simpleDataGenerator(ldays, 0.25, 5, 100, 5)
    for d in gen:
        print(d[0])

