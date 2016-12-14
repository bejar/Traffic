"""
.. module:: MyRemoteMonitor

MyRemoteMonitor
*************

:Description: MyRemoteMonitor

    

:Authors: bejar
    

:Version: 

:Created on: 12/12/2016 15:37 

"""

__author__ = 'bejar'

from keras.callbacks import Callback
import json


class MyRemoteMonitor(Callback):
    '''Callback used to stream events to a server.
    Requires the `requests` library.
    # Arguments
        root: root url to which the events will be sent (at the end
            of every epoch). Events are sent to
            `root + '/publish/epoch/end/'` by default. Calls are
            HTTP POST, with a `data` argument which is a
            JSON-encoded dictionary of event data.
    '''

    def __init__(self,
                 id = '',
                 root='http://localhost:9000',
                 path='/publish/epoch/end/',
                 field='data'):
        super(Callback, self).__init__()
        self.id = id
        self.root = root
        self.path = path
        self.field = field

    def on_epoch_end(self, epoch, logs={}):
        import requests
        send = {}
        send['epoch'] = epoch
        send['id'] = self.id
        for k, v in logs.items():
            send[k] = v
        print(send)
        try:
            requests.post(self.root + self.path,
                          params=send)
        except:
            print('Warning: could not reach RemoteMonitor '
                  'root server at ' + str(self.root))
