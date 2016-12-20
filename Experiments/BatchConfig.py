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


__author__ = 'bejar'

if __name__ == '__main__':

    ldaysTr = ['20161102','20161103','20161104','20161105','20161106','20161107','20161108','20161109','20161110',
               '20161111', '20161112', '20161113', '20161114', '20161115', '20161116', '20161117', '20161118',
               '20161119', '20161120', '20161121', '20161122', '20161123']
    ldaysTs = ['20161124']
    z_factor = 0.25

    smodel = 3
    classweight = {'0': 1.5, '1': 1, '2': 2.0, '3': 3.0, '4': 4.0}

    config = {'train': ldaysTr,
              'test': ldaysTs,
              'zfactor': 0.25,
              'model': smodel,
              'dpconvo': 0.2,
              'dpfull': 0.7,
              'convofields': [3, 3],
              'fulllayers': [64, 32],
              'classweight': classweight,
              'epochs': 100,
              'lrate': 0.005,
              'decay': 0.005/100,
              'batchsize': 100,
              'momentum': 0.9}



    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    host = 'itza'
    bconfig = {'done': False,
               'host': host,
               'config': config}

    col.insert(bconfig)

