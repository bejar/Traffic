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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import base64
import seaborn as sns
import numpy as np
from Traffic.Private.DBConfig import mongoconnection
import pprint
import time


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

    vals = col.find({'done': False}, {'_id':1,'acc':1, 'loss': 1, 'val_acc':1, 'val_loss':1, 'host':1, 'time_upd':1, 'time_init': 1, 'config':1})

    res = {}
    for v in vals:
        if len(v['acc']) > 0:

            # if we are resuming a stopped training we have to discount the epochs of the previous training to
            # compute the end time
            if 'epochs_trained' not in v['config']['train']:
                epochdiscount = 0
            else:
                epochdiscount = v['config']['train']['epochs_trained']

            tminit = time.mktime(time.strptime(v['time_init'], '%Y-%m-%d %H:%M:%S'))
            tmupd = time.mktime(time.strptime(v['time_upd'], '%Y-%m-%d %H:%M:%S')) 
            
            tepoch = ((tmupd-tminit)/ (len(v['acc']) - epochdiscount))
            ep = np.sum(v['config']['train']['epochs']) - len(v['acc'])
            id = int(tmupd+(tepoch*ep))

            # id is the approximated end time in seconds, so the table will be sorted that way
            res[id] = {}
            res[id]['id'] = v['_id']
            res[id]['epoch'] = len(v['acc'])
            res[id]['acc'] = v['acc'][-1]
            res[id]['val_acc'] = v['val_acc'][-1]
            res[id]['host'] = v['host']
            res[id]['init'] = time.strftime('%m/%d %H:%M:%S', time.localtime(tminit))
            res[id]['upd'] = time.strftime('%m/%d %H:%M:%S', time.localtime(tmupd))
            res[id]['end'] = time.strftime('%m/%d %H:%M:%S', time.localtime(tmupd+(tepoch*ep)))


            res[id]['eptime'] = ((tmupd-tminit)/ (len(v['acc']))) /60.0

            if len(v['acc']) >1:
                res[id]['acc_dir'] = v['acc'][-1] > v['acc'][-2]
                res[id]['val_acc_dir'] = v['val_acc'][-1] > v['val_acc'][-2]
            else:
                res[id]['acc_dir'] = True
                res[id]['val_acc_dir'] = True



    vals = col.find({'done': True, 'final_val_acc': {'$gt': 0.7}},
                    {'_id': 1,'final_acc': 1, 'final_val_acc': 1, 'val_loss': 1})

    old = {}

    for v in vals:
        res[v['_id']] = {}
        res[v['_id']]['final_acc'] = v['final_acc']
        res[v['_id']]['final_val_acc'] = v['final_val_acc']


    return render_template('Monitor.html', data=res, old=old)


@app.route('/Logs')
def logs():
    """
    Returns the logs in the DB
    """
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    vals = col.find({},  {'final_acc':1, 'final_val_acc':1, 'time_init': 1, 'time_end': 1, 'time_upd':1, 'acc':1,'done':1, 'mark':1, 'config':1})
    res = {}
    for v in vals:
        if 'time_init' in v:
            res[v['_id']] = {}
            if 'mark' in v:
                res[v['_id']]['mark'] = v['mark']
            else:
                res[v['_id']]['mark'] = False
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
            res[v['_id']]['zfactor'] = v['config']['zfactor']

    return render_template('Logs.html', data=res)

@app.route('/Mark', methods=['GET','POST'])
def mark():
    """
    Marks an experiment
    :return:
    """
    payload = request.form['mark']
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]
    vals = col.find_one({'_id': int(payload)},  {'mark':1, 'done':1})

    text = ' Not Marked'
    if vals['done']:
        if not 'mark' in vals:
            marked = True
        else:
            marked = not vals['mark']

        col.update({'_id':vals['_id']},{'$set': {'mark': marked}})
        text = ' Marked'

    head = """
    <!DOCTYPE html>
<html>
<head>
    <title>Keras NN Mark </title>
   <meta http-equiv="refresh" content="3;http://%s:%d/Logs" />
  </head>
<body>
""" % (hostname, port)
    end = '</body></html>'


    return head + str(payload) + text + end


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

    lstyles = ['-', '-', '-', '-'] *3
    lcolors = ['r', 'g', 'b', 'y'] *3
    payload = request.form['graph']

    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]

    vals = col.find_one({'_id': int(payload)}, {'acc':1, 'loss': 1, 'val_acc':1, 'val_loss':1})
    if vals is not None:
        del vals['_id']

        img = StringIO.StringIO()

        fig = plt.figure(figsize=(10,8),dpi=200)
        axes = fig.add_subplot(1,1,1)

        for v, color, style in zip(sorted(vals), lcolors, lstyles):
            axes.plot(range(len(vals[v])),vals[v], color + style, label=v)

        axes.set_xlabel('epoch')
        axes.set_ylabel('acc/loss')
        axes.set_title("Training/Test")
        axes.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        axes.xaxis.set_major_locator(ticker.MultipleLocator(25))

        plt.legend()
        plt.savefig(img, format='png')
        img.seek(0)

        plot_url = base64.b64encode(img.getvalue())
        plt.close()

        return render_template('Graphview.html', plot_url=plot_url, acc=vals['acc'][-1], vacc=vals['val_acc'][-1], id=payload, ep=len(vals['acc']))
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
           '<br><h2>Config:</h2><br><br>' + pprint.pformat(vals['config'], indent=4, width=60).replace('\n', '<br>') + \
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


@app.route('/Stop', methods=['GET','POST'])
def stop():
    """
    Writes on the DB configuration of the process that it has to stop the next epoch

    :return:
    """
    payload = request.form['stop']

    print payload
    client = MongoClient(mongoconnection.server)
    db = client[mongoconnection.db]
    db.authenticate(mongoconnection.user, password=mongoconnection.passwd)
    col = db[mongoconnection.col]
    vals = col.find_one({'_id': int(payload)}, {'host':1, 'stop':1})
    print vals
    print col.update({'_id':int(payload)}, {'$set': {'stop': True}})

    head = """
    <!DOCTYPE html>
<html>
<head>
    <title>Keras NN Stop </title>
   <meta http-equiv="refresh" content="3;http://%s:%d/Monitor" />
  </head>
<body>
""" % (hostname, port)
    end = '</body></html>'


    return head + str(payload) + ' Stopped' + end


if __name__ == '__main__':
    # The Flask Server is started
    app.run(host='0.0.0.0', port=port, debug=False)
