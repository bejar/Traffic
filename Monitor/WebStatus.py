"""
.. module:: WebStatus

ConvoTest
*************

:Description: WebStatus



:Authors: bejar


:Version:

:Created on: 28/11/2016 11:10

"""

__author__ = 'bejar'

import socket
import time


from flask import Flask, request, render_template
#from Parameters.Private import WS_port


# Configuration stuff
hostname = socket.gethostname()
port = 8850

app = Flask(__name__)

acc = {}
loss = {}
val_acc = {}
val_loss = {}
status = {}
epoch = {}

@app.route("/Update", methods=['GET','POST'])
def update():
    """
    Updates the status of the process collecting tweets from a city
    @return:
    """
    global acc
    global val_acc
    global status
    global epoch

    strtime = time.ctime(int(time.time()))
    try:
        id = request.args['id']
        acc[id] = []
        loss[id] = []
        val_acc[id] = []
        val_loss[id] = []
        status[id] = strtime
        if 'epoch' in request.args:
            epoch[id] += request.args['epoch']
        if 'acc' in request.args:
            acc[id] += request.args['acc']
        if 'val_acc' in request.args:
            val_acc[id] += request.args['val_acc']
    except:
        return 'KO'

    return 'Ok'


@app.route('/Status')
def info():
    """
    Status de las ciudades
    """
    global acc
    global loss
    global val_acc
    global val_loss
    global status
    global epoch

    return render_template('Status.html', status=status, acc=acc,  val_acc=val_acc, epoch=epoch)


if __name__ == '__main__':
    # Ponemos en marcha el servidor Flask
    app.run(host='0.0.0.0', port=port, debug=False)
