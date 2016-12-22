"""
.. module:: BatchConfig

BatchConfig
*************

:Description: BatchConfig

    stores in the database configs so can be retrieved by a training process

:Authors: bejar
    

:Version: 

:Created on: 20/12/2016 14:26 

"""

from pymongo import MongoClient
from Util.DBConfig import mongoconnection
from Util.Generate_Dataset import list_days_generator
import time

__author__ = 'bejar'

if __name__ == '__main__':

    ldaysTr = list_days_generator(2016, 11, 1, 23)
    ldaysTs = list_days_generator(2016, 11, 24, 24)
    z_factor = 0.25

    smodel = 3
    classweight = {'0': 1.5, '1': 1, '2': 2.0, '3': 3.0, '4': 4.0}

    config = {'train': ldaysTr,
              'test': ldaysTs,
              'zfactor': 0.25,
              'model': smodel,
              'dpconvo': 0.2,
              'dpfull': 0.6,
              'convofields': [3, 3],
              'fulllayers': [64, 32],
              'classweight': classweight,
              'epochs': 100,
              'lrate': 0.005,
              'decay': 0.005/100,
              'batchsize': 100,
              'momentum': 0.9}

    config['generator'] = False
    config['samples_epoch'] = 50000

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    host = 'itza'
    bconfig = {'_id': int(time.time()),
               'pending': True,
               'host': host,
               'config': config}

    col.insert(bconfig)

