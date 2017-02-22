"""
.. module:: DBLog

DBLog
*************

:Description: DBLog

    Logs the training of a network saving log info in a MongoDB,
    if connection fails at any epoch log info is not saved in the database
    if connection fails at the end of training log is saved to a json file

:Authors: bejar
    

:Version: 

:Created on: 16/12/2016 8:29 

"""

__author__ = 'bejar'

from keras.callbacks import Callback
import time
from pymongo import MongoClient
import socket
from keras.utils.visualize_util import model_to_dot
from pymongo.errors import ConnectionFailure
import json
import numpy as np
from numpy.random import randint

class DBLog(Callback):
    """
    Callback used to stream events to a DB
    """

    def __init__(self, database, config, model, modelj, resume=None):
        super(Callback, self).__init__()

        self.mgdb = database
        self.config = config

        if resume is None:
            self.id = int(time.time()) + randint(0, 50)
            svgmodel = model_to_dot(model, show_shapes=True).create(prog='dot', format='svg')
            self.backup = {'_id': self.id,
                        'host': socket.gethostname().split('.')[0],
                        'model': modelj,
                        'svgmodel': svgmodel,
                        'config': self.config,
                        'acc': [],
                        'loss': [],
                        'val_acc': [],
                        'val_loss': [],
                        'time_init': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                        'time_upd': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                        'done': False
                        }

            try: # Try to save log in DB
                client = MongoClient(self.mgdb.server)
                db = client[self.mgdb.db]
                db.authenticate(self.mgdb.user, password=self.mgdb.passwd)
                col = db[self.mgdb.col]
                col.insert(self.backup)
            except ConnectionFailure:
                pass
        else:
            self.id = config['_id']
            self.backup = resume
            self.backup['host'] = socket.gethostname().split('.')[0]
            self.backup['time_init'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            self.backup['time_upd'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

            try: # Try to save log in DB
                client = MongoClient(self.mgdb.server)
                db = client[self.mgdb.db]
                db.authenticate(self.mgdb.user, password=self.mgdb.passwd)
                col = db[self.mgdb.col]
                col.update({'_id':self.id}, {'$set': {'host': self.backup['host']}})
                col.update({'_id':self.id}, {'$set': {'time_init': self.backup['time_init']}})
                col.update({'_id':self.id}, {'$set': {'time_upd': self.backup['time_upd']}})
            except ConnectionFailure:
                pass


    def on_epoch_end(self, epoch, logs={}):

        for k, v in logs.items():
            self.backup[k].append(v)
        self.backup['time_upd'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

        try: # Try to save log in DB
            client = MongoClient(self.mgdb.server)
            db = client[self.mgdb.db]
            db.authenticate(self.mgdb.user, password=self.mgdb.passwd)
            col = db[self.mgdb.col]

            exists = col.find_one({'_id':self.id}, {'done':1})
            if exists is not None:
                col.update({'_id':self.id}, {'$set': {'acc': self.backup['acc']}})
                col.update({'_id':self.id}, {'$set': {'loss': self.backup['loss']}})
                col.update({'_id':self.id}, {'$set': {'val_loss': self.backup['val_loss']}})
                col.update({'_id':self.id}, {'$set': {'val_acc': self.backup['val_acc']}})
                col.update({'_id':self.id}, {'$set': {'time_upd': self.backup['time_upd']}})
            else:
                col.insert(self.backup)

        except ConnectionFailure:
            pass

    def on_train_end(self, logs={}):

        self.backup['done'] = True
        self.backup['time_end'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.backup['final_acc'] = logs['acc']
        self.backup['final_val_acc'] = logs['val_acc']

        try: # Try to save log in DB
            client = MongoClient(self.mgdb.server)
            db = client[self.mgdb.db]
            db.authenticate(self.mgdb.user, password=self.mgdb.passwd)
            col = db[self.mgdb.col]

            exists = col.find_one({'_id':self.id}, {'done':1})
            if exists is not None:
                col.update({'_id':self.id}, {'$set': {'done': self.backup['done'],
                                                      'time_end': self.backup['time_end'],
                                                      'final_acc': self.backup['final_acc'] ,
                                                      'final_val_acc': self.backup['final_val_acc'],
                                                      }})
            else:
                col.insert(self.backup)

        except ConnectionFailure:
            pass

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

        try: # Try to save log in DB
            client = MongoClient(self.mgdb.server)
            db = client[self.mgdb.db]
            db.authenticate(self.mgdb.user, password=self.mgdb.passwd)
            col = db[self.mgdb.col]

            exists = col.find_one({'_id':self.id}, {'done':1})
            if exists is not None:
                col.update({'_id':self.id}, {'$set': {'confusion': self.backup['confusion'],
                                                      'report': self.backup['report'],
                                                      'accuracy': self.backup['accuracy']
                                                      }})
            else:
                col.insert(self.backup)

        except ConnectionFailure:
            with open(self.backup['savepath'] + '/' + str(self.id) + '.json', 'w') as outfile:
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