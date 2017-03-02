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
from Traffic.Util.Misc import list_days_generator
from collections import Counter

def name_days_file(ldays):
    """
    Generates a name file using the first and last day from the dataset
    It is assumed that ldays is the list of training days from the configuration file
    :param ldays:
    :return:
    """

    p = []
    for year, month, iday, fday in ldays:
        p.extend(list_days_generator(year, month, iday, fday))
    return p[0] + '-' + p[-1]


class Dataset:

    def __init__(self, datapath, ldays, zfactor, imgord='th', nclasses=None, recode=None):
        """
        Checks if the file exists

        :param datapath:
        :param days:
        :param zfactor:
        :param nclases:
        :param merge: Merge classes
        """
        self.fname = datapath + '/' + "Data-" + name_days_file(ldays) + '-Z%0.2f-%s' % (zfactor, imgord) + '.hdf5'
        self.recode = recode
        self.labprop = None
        self.X_train = None
        self.y_labels = None
        self.input_shape = None
        self.chunk_size = None
        self.chunks = None

        if not os.path.isfile(self.fname):
            raise Exception('Data file does not exists')
        self.handle = None

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

            if self.recode is None:
                self.y_labels = y_train
            else:
                self.y_labels = [self.recode[i] for i in y_train]
                self.nclasses = len(np.unique(self.y_labels))

            cnt = Counter(self.y_labels)
            lprop = {}
            for l in cnt:
                # String key because is going to be converted to JSON
                lprop[str(l)] = cnt[l]/float(len(self.y_labels))

            self.labprop = lprop
            self.y_train = np_utils.to_categorical(self.y_labels, self.nclasses)
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
            if self.X_train.shape[0] % batchsize != 0:
                raise Exception('Chunksize not a multiple of batchsize')
            if self.recode is None:
                tlabels = self.handle[chunk + '/labels'][()]
            else:
                tlabels = [self.recode[i] for i in self.handle[chunk + '/labels'][()]]
            self.y_train = np_utils.to_categorical(tlabels, self.nclasses)

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

    def describe(self):
        """
        Prints information about the dataset object

        :return:
        """

        print (self.fname)
        if self.handle is not None:
            print("File Open")
        else:
            print("File Closed")
        print ('NC= %d' % len(self.chunks))
        print ('CS= %d' % self.chunk_size)

        if self.recode is not None:
            print('RC = %s' % self.recode)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    d = Dataset(process_path, [[2016, 11, 1, 30]], 0.25, nclasses=5)
    d.open()
    chunks, _ = d.chunks()
    # for chunk in chunks:
    #     d.load_chunk(chunk, 256)
    #     print d.perm
    # d.in_memory()
    d.load_chunk(chunks[0], 128)

    print (d.X_train[0].transpose((1,2,0))).shape

    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(10)
    sp1 = fig.add_subplot(1,1,1)
    sp1.imshow(d.X_train[100].transpose((1,2,0)))
    plt.show()
    plt.close()

    d.close()