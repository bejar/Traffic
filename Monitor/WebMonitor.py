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

from flask import Flask, render_template, request
from pymongo import MongoClient

import StringIO

import bokeh.plotting as plt

import matplotlib
matplotlib.use('Agg')

import  matplotlib.pyplot as plt
import base64
import seaborn as sns

from Util.DBConfig import mongoconnection
import pprint


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

    vals = col.find({'done': False}, {'_id':1,'acc':1, 'loss': 1, 'val_acc':1, 'val_loss':1, 'host':1, 'time_upd':1, 'time_init': 1})

    res = {}
    for v in vals:
        if len(v['acc']) > 0:
            res[v['_id']] = {}
            res[v['_id']]['epoch'] = len(v['acc'])
            res[v['_id']]['acc'] = v['acc'][-1]
            res[v['_id']]['val_acc'] = v['val_acc'][-1]
            res[v['_id']]['host'] = v['host']
            res[v['_id']]['upd'] = v['time_upd']
            res[v['_id']]['init'] = v['time_init']
            if len(v['acc']) >1:
                res[v['_id']]['acc_dir'] = v['acc'][-1] > v['acc'][-2]
                res[v['_id']]['val_acc_dir'] = v['val_acc'][-1] > v['val_acc'][-2]
            else:
                res[v['_id']]['acc_dir'] = True
                res[v['_id']]['val_acc_dir'] = True



    vals = col.find({'done': True, 'final_val_acc': {'$gt': 0.7}},
                    {'_id': 1,'final_acc': 1, 'final_val_acc': 1, 'val_loss': 1})

    old = {}

    for v in vals:
        res[v['_id']] = {}
        res[v['_id']]['final_acc'] = v['final_acc']
        res[v['_id']]['final_val_acc'] = v['final_val_acc']


    return render_template('Monitor.html', data=res, old=old)

@app.route('/Batch')
def batch():
    """
    Returns the batches pending in the DB
    """
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    vals = col.find({'pending': True})
    res = {}
    for v in vals:
        res[v['_id']] = {}
        res[v['_id']]['host'] = v['host']
    return render_template('Batch.html', data=res)

@app.route('/Logs')
def logs():
    """
    Returns the logs in the DB
    """
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    vals = col.find({},  {'final_acc':1, 'final_val_acc':1, 'time_init': 1, 'time_end': 1, 'done':1})
    res = {}
    for v in vals:
        if 'time_init' in v:
            res[v['_id']] = {}
            if 'final_acc' in v:
                res[v['_id']]['acc'] = v['final_acc']
            else:
                res[v['_id']]['acc'] = 0
            if 'final_val_acc' in v:
                res[v['_id']]['val_acc'] = v['final_val_acc']
            else:
                res[v['_id']]['val_acc'] = 0
            res[v['_id']]['init'] = v['time_init']
            if 'time_end' in v:
                res[v['_id']]['end'] = v['time_end']
            else:
                 res[v['_id']]['end'] = 'pending'

    return render_template('Logs.html', data=res)

@app.route('/Delete', methods=['GET','POST'])
def delete():
    """
    Deletes a log
    """
    payload = request.form['delete']
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    col.remove({'_id': int(payload)})

    head = """
    <!DOCTYPE html>
<html>
<head>
    <title>Keras NN Delete </title>
   <meta http-equiv="refresh" content="3;http://%s:%d/Logs" />
  </head>
<body>
""" % (hostname, port)
    end = '</body></html>'


    return head + str(payload) + ' Removed' + end

@app.route('/Graph', methods=['GET','POST'])
def graphic():
    """
    Generates a page with the training trace

    :return:
    """

    lstyles = ['-', '-', ':', ':'] *3
    payload = request.form['graph']

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    vals = col.find_one({'_id': int(payload)}, {'acc':1, 'loss': 1, 'val_acc':1, 'val_loss':1})
    if vals is not None:
        del vals['_id']

        img = StringIO.StringIO()

        fig = plt.figure(figsize=(5,4),dpi=100)
        axes = fig.add_subplot(1,1,1)

        for v,lstyle in zip(sorted(vals), lstyles):
            axes.plot(range(len(vals[v])),vals[v],lstyle, label=v)

        axes.set_xlabel('epoch')
        axes.set_ylabel('acc/loss')
        axes.set_title("Training/Test")

        plt.legend()
        plt.savefig(img, format='png')
        img.seek(0)

        plot_url = base64.b64encode(img.getvalue())
        plt.close()

        return render_template('graphview.html', plot_url=plot_url)
    else:
        return ""


@app.route('/Model', methods=['GET','POST'])
def model():
    """
    Generates a page with the configuration of the training and the model

    :return:
    """


    payload = request.form['model']

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    vals = col.find_one({'_id': int(payload)}, {'model':1, 'config':1, 'svgmodel':1})
    pp = pprint.PrettyPrinter(indent=4)

    if 'svgmodel' in vals:
        svgmodel = vals['svgmodel']
    else:
        svgmodel = ''

    head = """
    <!DOCTYPE html>
<html>
<head>
    <title>Keras NN Config </title>
  </head>
<body>
"""
    end = '</body></html>'

    return head + \
           '<br><h2>Config:</h2><br><br>' + pprint.pformat(vals['config'], indent=4, width=40).replace('\n', '<br>') + \
           '<br><br><h2>Graph:</h2><br><br>' + svgmodel +'<br><br><h2>Net:</h2><br><br>'+ \
           pprint.pformat(vals['model'], indent=4, width=40).replace('\n', '<br>') + \
            '<br>' + \
           end


@app.route('/Report', methods=['GET','POST'])
def report():
    """
    Returns a web page with the classification report

    :return:
    """
    payload = request.form['model']

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    vals = col.find_one({'_id': int(payload)}, {'report':1, 'confusion':1})

    head = """
    <!DOCTYPE html>
<html>
<head>
    <title>Keras NN Config </title>
  </head>
<body>
"""
    end = '</body></html>'

    if 'report' in vals:
        return head + \
               '<br><h2>Report:</h2><pre>' + vals['report'] + \
               '</pre><br><br><h2>Confusion:</h2><pre>' + vals['confusion'] +'</pre><br><br>' + \
               end

    else:
        return 'No report'


@app.route('/BConfig', methods=['GET','POST'])
def config():
    """
    Generates a page with the configuration of the batch

    :return:
    """
    payload = request.form['config']

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    vals = col.find_one({'_id': int(payload)}, {'config':1})
    pp = pprint.PrettyPrinter(indent=4)

    head = """
    <!DOCTYPE html>
<html>
<head>
    <title>Keras NN Config </title>
  </head>
<body>
"""
    end = '</body></html>'

    return head + \
           '<br><h2>Config:</h2><br><br>' + pprint.pformat(vals['config'], indent=4, width=40).replace('\n', '<br>') + \
           end


if __name__ == '__main__':
    # Ponemos en marcha el servidor Flask
    app.run(host='0.0.0.0', port=port, debug=False)
