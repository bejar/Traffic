"""
.. module:: WebMonitor

ConvoTest
*************

:Description: WebStatus



:Authors: bejar


:Version:

:Created on: 28/11/2016 11:10

"""

import socket

from flask import Flask, render_template
from pymongo import MongoClient

from Util.DBConfig import mongoconnection
#from Parameters.Private import WS_port

__author__ = 'bejar'

# Configuration stuff
hostname = socket.gethostname()
port = 8850

app = Flask(__name__)




@app.route('/Monitor')
def info():
    """
    Status de las ciudades
    """

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    vals = col.find({'done': False}, {'_id':1,'acc':1, 'loss': 1, 'val_acc':1, 'val_loss':1, 'host':1})

    res = {}
    for v in vals:
        if len(v['acc'])>0:
            res[v['_id']] = {}
            res[v['_id']]['epoch'] = len(v['acc'])
            res[v['_id']]['acc'] = v['acc'][-1]
            res[v['_id']]['val_acc'] = v['val_acc'][-1]
            res[v['_id']]['host'] = v['host']

    return render_template('Monitor.html', data=res)

@app.route('/Graph')
def graphic():
    """
    Generates a page with the training trace

    :return:
    """

if __name__ == '__main__':
    # Ponemos en marcha el servidor Flask
    app.run(host='0.0.0.0', port=port, debug=False)
