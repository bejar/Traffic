"""
.. module:: DBLog

DBLog
*************

:Description: DBLog

    

:Authors: bejar
    

:Version: 

:Created on: 16/12/2016 8:29 

"""

__author__ = 'bejar'


from keras.callbacks import Callback
import time
from pymongo import MongoClient
import socket
import pydotplus as pydot
import pickle
from keras.utils.visualize_util import model_to_dot

class DBLog(Callback):
    '''Callback used to stream events to a DB
    '''

    def __init__(self, database, config, model, modelj):
        super(Callback, self).__init__()
        self.id = int(time.time())
        self.mgdb = database
        self.config = config
        client = MongoClient(self.mgdb.server)
        db = client[self.mgdb.db]
        db.authenticate(self.mgdb.user, password=self.mgdb.passwd)
        col = db[self.mgdb.col]
        dotobj = model_to_dot(model)

        col.insert({'_id': self.id,
                    'host': socket.gethostname().split('.')[0],
                    'model': modelj,
                    'config': self.config,
                    'acc': [],
                    'loss': [],
                    'val_acc': [],
                    'val_loss': [],
                    'time_init': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                    'time_upd': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                    'dotobj': pickle.dumps(dotobj),
                    'done': False
                    })

    def on_epoch_end(self, epoch, logs={}):
        client = MongoClient(self.mgdb.server)
        db = client[self.mgdb.db]
        db.authenticate(self.mgdb.user, password=self.mgdb.passwd)
        col = db[self.mgdb.col]

        send = col.find_one({'_id':self.id}, {'acc':1, 'loss': 1, 'val_acc':1, 'val_loss':1})
        for k, v in logs.items():
            send[k].append(v)

        col.update({'_id':self.id}, {'$set': {'acc': send['acc']}})
        col.update({'_id':self.id}, {'$set': {'loss': send['loss']}})
        col.update({'_id':self.id}, {'$set': {'val_loss': send['val_loss']}})
        col.update({'_id':self.id}, {'$set': {'val_acc': send['val_acc']}})
        col.update({'_id':self.id}, {'$set': {'time_upd': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}})


    def on_train_end(self, logs={}):

        client = MongoClient(self.mgdb.server)
        db = client[self.mgdb.db]
        db.authenticate(self.mgdb.user, password=self.mgdb.passwd)
        col = db[self.mgdb.col]
        send = col.find_one({'_id':self.id}, {'acc':1, 'val_acc':1})
        col.update({'_id':self.id}, {'$set': {'done':True,
                                              'time_end':time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                              'final_acc': send['acc'][-1],
                                              'final_val_acc': send['val_acc'][-1],
                                              }})

    def save_final_results(self, accuracy, confusion, report):
        """
        Adds  accuracy, confusion matrix and classification report to the DB
        :param confusion:
        :param report:
        :return:
        """
        client = MongoClient(self.mgdb.server)
        db = client[self.mgdb.db]
        db.authenticate(self.mgdb.user, password=self.mgdb.passwd)
        col = db[self.mgdb.col]

        sconfusion = ""
        for i1 in range(confusion.shape[0]):
            for i2 in range(confusion.shape[1]):
                sconfusion += "%4d " % confusion[i1,i2]
            sconfusion += "\n"


        col.update({'_id':self.id}, {'$set': {'confusion': sconfusion, 'report': report, 'accuracy': accuracy}})