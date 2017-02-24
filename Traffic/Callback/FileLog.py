"""
.. module:: DBLog

DBLog
*************

:Description: FileLog

    Logs the training of a network saving log info into a .json file,

:Authors: bejar


:Version:

:Created on: 16/12/2016 8:29

"""

__author__ = 'bejar'

from keras.callbacks import Callback
import time
import socket
import json
import numpy as np
from numpy.random import randint


class FileLog(Callback):
    """
    Callback used to stream events to a DB
    """

    def __init__(self, config, modelj):
        super(Callback, self).__init__()

        self.id = int(time.time()) + randint(0, 50)
        self.backup = {'_id': self.id,
                       'host': socket.gethostname().split('.')[0],
                       'config': config,
                       'model': modelj,
                     '  acc': [],
                       'loss': [],
                       'val_acc': [],
                       'val_loss': [],
                       'time_init': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                       'time_upd': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                       'done': False
                       }

        with open(self.backup['config']['savepath']+ '/' + str(self.id) + '.json', 'w') as outfile:
            json.dump(self.backup, outfile)

    def on_epoch_end(self, epoch, logs={}):

        for k, v in logs.items():
            self.backup[k].append(v)
        self.backup['time_upd'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

        with open(self.backup['config']['savepath'] + '/' + str(self.id) + '.json', 'w') as outfile:
            json.dump(self.backup, outfile)

    def on_train_end(self, logs={}):

        self.backup['done'] = True
        self.backup['time_end'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.backup['final_acc'] = logs['acc']
        self.backup['final_val_acc'] = logs['val_acc']

        with open(self.backup['config']['savepath'] + '/' + str(self.id) + '.json', 'w') as outfile:
            json.dump(self.backup, outfile)

    def save_final_results(self, accuracy, confusion, report):
        """
        Adds  accuracy, confusion matrix and classification report to the DB
        :param confusion:
        :param report:
        :return:
        """
        sconfusion = ""
        for i1 in range(confusion.shape[0]):
            for i2 in range(confusion.shape[1]):
                sconfusion += "%4d " % confusion[i1, i2]
            sconfusion += "\n"

        self.backup['confusion'] = sconfusion
        self.backup['report'] = report
        self.backup['accuracy'] = accuracy

        with open(self.backup['config']['savepath'] + '/' + str(self.id) + '.json', 'w') as outfile:
            json.dump(self.backup, outfile)

    def is_best_epoch(self):
        """
        Returns if the last accuracy for the TEST data is the highest accuracy value
        Useful for deciding when to save the trained models

        :return:
        """

        if len(self.config['val_acc']) > 1:
            return(self.config['val_acc'][-1] > np.max(self.config['val_acc'][:-1]))
        else:
            return True