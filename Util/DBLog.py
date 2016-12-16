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



class DBLog(Callback):
    '''Callback used to stream events to a DB
    '''

    def __init__(self, id, database, config):
        super(Callback, self).__init__()
        self.id = int(time.time())
        self.mgdb = database
        self.config = config
        client = MongoClient(self.mgdb.server)
        db = client[self.mgdb.db]
        col = db[self.mgdb.col]
        col.insert({'_id':self.id,
                    'config':self.config,
                    'epoch':[],
                    'acc':[],
                    'loss':[],
                    'val_acc':[],
                    'val_loss':[],
                    'time_init': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                    'done': False
                    })

    def on_epoch_end(self, epoch, logs={}):
        client = MongoClient(self.mgdb.server)
        db = client[self.mgdb.db]
        col = db[self.mgdb.col]

        send = col.find_one({'_id':self.id}, {'acc':1, 'loss': 1, 'val_acc':1, 'val_loss':1, 'epoch':1})
        for k, v in logs.items():
            send[k] += v
        try:
            col.update({'_id':self.id}, {'$set': send})

        except:
            print('Warning: could not reach DB')


    def on_train_end(self, logs={}):

        client = MongoClient(self.mgdb.server)
        db = client[self.mgdb.db]
        col = db[self.mgdb.col]

        col.update({'_id':self.id}, {'$set': {'Done':True, 'time_end':time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}})
