"""
.. module:: DirMonitor

ConvoTest
*************

:Description: DirMonitor

 Monitors the training files in a directory

 Mimicks WebMonitoy but without a database


:Authors: bejar


:Version:

:Created on: 28/11/2016 11:10

"""

import socket
import glob
from flask import Flask, render_template, request
import StringIO


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import base64
import seaborn as sns
import numpy as np
import pprint
import time
import argparse
import json

from Traffic.Util.Misc import load_config_file


__author__ = 'bejar'

# Configuration stuff
hostname = socket.gethostname()
port = 8850

app = Flask(__name__)

def retrieve_all(datafiles):
    """
    Retrieves all .json files
    :return:
    """

    return

@app.route('/Monitor')
def info():
    """
    Status de las ciudades
    """

    vals = [load_config_file(f, abs=True) for f in glob.glob(datafiles + '/*.json')]

    res = {}
    for v in vals:
        id = v['_id']
        res[id] = {}
        res[id]['id'] = v['_id']
        res[id]['zfactor'] = v['config']['zfactor']
        res[id]['epoch'] = len(v['acc'])
        tminit = time.mktime(time.strptime(v['time_init'], '%Y-%m-%d %H:%M:%S'))
        res[id]['init'] = time.strftime('%m/%d %H:%M:%S', time.localtime(tminit))

        if 'done' in v:
            res[id]['done'] = v['done']
        else:
            res[id]['done'] = False

        if len(v['acc']) >0:
            tmupd = time.mktime(time.strptime(v['time_upd'], '%Y-%m-%d %H:%M:%S'))
            tepoch = ((tmupd-tminit)/ (len(v['acc'])))
            ep = np.sum(v['config']['train']['epochs']) - len(v['acc'])
            res[id]['acc'] = v['acc'][-1]
            res[id]['val_acc'] = v['val_acc'][-1]
            res[id]['upd'] = time.strftime('%m/%d %H:%M:%S', time.localtime(tmupd))
            res[id]['end'] = time.strftime('%m/%d %H:%M:%S', time.localtime(tmupd + (tepoch * ep)))
            res[id]['eptime'] = ((tmupd-tminit)/ (len(v['acc']))) /60.0
        else:
            res[id]['acc'] = 0
            res[id]['val_acc'] = 0
            res[id]['upd'] = "pending"
            res[id]['end'] = "pending"
            res[id]['eptime'] = 0

        if len(v['acc']) >1:
            res[id]['acc_dir'] = v['acc'][-1] > v['acc'][-2]
            res[id]['val_acc_dir'] = v['val_acc'][-1] > v['val_acc'][-2]
        else:
            res[id]['acc_dir'] = True
            res[id]['val_acc_dir'] = True

    old = {}


    return render_template('FMonitor.html', data=res, old=old)





@app.route('/Graph', methods=['GET','POST'])
def graphic():
    """
    Generates a page with the training trace

    :return:
    """

    lstyles = ['-', '-', '-', '-'] *3
    lcolors = ['r', 'g', 'b', 'y'] *3
    payload = request.form['graph']

    vals = load_config_file(datafiles + payload + '.json', abs=True)
    if vals is not None:
        nvals = ['acc', 'val_acc', 'loss', 'val_loss']

        img = StringIO.StringIO()

        fig = plt.figure(figsize=(10,8),dpi=200)
        axes = fig.add_subplot(1,1,1)

        for v, color, style in zip(sorted(nvals), lcolors, lstyles):
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

    vals = load_config_file(datafiles + payload + '.json', abs=True)

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
           '<br><h2>Config:</h2><br><br>' + pprint.pformat(vals['config'], indent=4, width=60).replace('\n', '<br>') + \
           '<br><br><h2>Graph:</h2><br><br>'  +'<br><br><h2>Net:</h2><br><br>'+ \
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

    vals = load_config_file(datafiles + payload + '.json', abs=True)

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


@app.route('/Stop', methods=['GET', 'POST'])
def stop():
    """
    Writes on the DB configuration of the process that it has to stop the next epoch

    :return:
    """
    payload = request.form['model']
    config = load_config_file(datafiles + '/' + payload + '.json', abs=True)
    config['stop'] = True

    with open(datafiles + '/' + payload + '.json', 'w') as outfile:
        json.dump(config, outfile)


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

    parser = argparse.ArgumentParser()
    parser.add_argument('--datafiles', default='', help='Directory of files to monitor')
    args = parser.parse_args()

    datafiles = args.datafiles
    # The Flask Server is started
    app.run(host='0.0.0.0', port=port, debug=False)
