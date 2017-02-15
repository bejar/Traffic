"""
.. module:: Dataset

Dataset
*************

:Description: Dataset

    Encapsulates a dataset that is in a HDF5 file.
    it can load it all in memory or get only a chunk from the file

:Authors: bejar
    

:Version: 

:Created on: 14/02/2017 12:46 

"""

__author__ = 'bejar'

import os.path

import h5py
import numpy as np
from Traffic.Config.Constants import process_path
from keras.utils import np_utils
from numpy.random import shuffle


def name_days_file(ldays):
    """
    Generates a name file using the first and last day from the dataset
    :param ldays:
    :return:
    """

    p = []
    for year, month, iday, fday in ldays:
        p.extend(list_days_generator(year, month, iday, fday))
    return p[0] + '-' + p[-1]


def list_days_generator(year, month, iday, fday):
    """
    Generates a list of days

    :param year:
    :param month:
    :param iday:
    :param fday:
    :return:
    """
    ldays = []
    for v in range(iday, fday + 1):
        ldays.append("%d%d%02d" % (year, month, v))
    return ldays

class Dataset:

    def __init__(self, datapath, ldays, zfactor, nclasses=None):
        """
        Checks if the file exists

        :param datapath:
        :param days:
        :param zfactor:
        :param nclases:
        """

        self.fname =  datapath + '/' + "Data-" + name_days_file(ldays) + '-Z%0.2f' % zfactor + '.hdf5'

        if not os.path.isfile(self.fname):
            raise Exception('Data file does not exists')
        self.hande = None

        # If the dataset is going to be loaded in batches we need the number of classes
        if nclasses is not None:
            self.nclasses = nclasses

    def open(self):
        """
        Opens the hdf5 file
        :return:
        """
        self.handle = h5py.File(self.fname, 'r')

    def close(self):
        """
        Closes the hdf5 file
        :return:
        """
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def in_memory(self):
        """
        Loads all the chunks in the datafile in memory
        :return:
        """
        if self.handle is not None:
            chunks = [c for c in self.handle]

            X_train = []
            y_train = []

            for chunk in chunks:
                X_train.append(self.handle[chunk + '/data'][()])
                y_train.extend(self.handle[chunk + '/labels'][()])

            self.nclasses = len(np.unique(y_train))
            self.X_train = np.concatenate(X_train)
            self.y_labels = y_train
            self.y_train = np_utils.to_categorical(y_train, self.nclasses)
            self.input_shape = self.X_train[0].shape
        else:
            raise Exception("Data file not open")

    def load_chunk(self, chunk, batchsize):
        """
        Loads a chunk from the file
        :param chunk:
        :return:
        """
        if self.nclasses is None:
            raise Exception('Number of classes not initialized')
        if self.handle is not None:
            self.X_train = self.handle[chunk + '/data'][()]
            tlabels = self.handle[chunk + '/labels'][()]
            self.y_train = np_utils.to_categorical(tlabels, self.nclasses)
            if self.X_train.shape[0] % batchsize != 0:
                raise Exception('Chunksize not a multiple of batchsize')

            perm = [i for i in range(self.X_train.shape[0])]  # so shuffle works on python 3
            shuffle(perm)
            lperm = []
            for i in range(0, len(perm), batchsize):
                gperm = []
                for j in range(batchsize):
                    if (i + j) < len(perm):
                        gperm.append(perm[i + j])
                lperm.append(sorted(gperm))
            self.perm = lperm
        else:
            raise Exception("Data file not open")

    def chunks(self):
        """
        Returns a list with the names of the chunks in the file and the size of the chunks
        :return:
        """
        if self.handle is not None:
            self.chunks = [c for c in self.handle]
            self.chunk_size = self.handle[self.chunks[0] + '/data'].shape[0]
        else:
            raise Exception("Data file not open")

        return self.chunks, self.chunk_size

if __name__ == '__main__':
    d = Dataset(process_path, [[2016, 12, 1, 2]], 0.25, nclasses=5)
    d.open()
    # chunks, _ = d.chunks()
    # for chunk in chunks:
    #     d.load_chunk(chunk, 256)
    #     print d.perm
    d.in_memory()
    print d.input_shape
    d.close()