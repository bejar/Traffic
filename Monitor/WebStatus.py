__author__ = 'bejar'

import socket
import time
import json

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

@app.route("/Update", methods=['POST'])
def update():
    """
    Updates the status of the process collecting tweets from a city
    @return:
    """
    global acc
    global loss
    global val_acc
    global val_loss
    global status

    id = request.form.get('id')
    payload = request.form.get('data')
    strtime = time.ctime(int(time.time()))
    status[id] = strtime
    acc[id] = 0.0
    loss[id] = 0.0
    val_acc[id] = 0.0
    val_loss[id] = 0.0
    try:
        data = json.loads(payload)
        if 'acc' in data:
            acc[id] = data['acc']
        if 'loss' in data:
            acc[id] = data['loss']
        if 'val_acc' in data:
            acc[id] = data['val_acc']
        if 'val_loss' in data:
            acc[id] = data['val_loss']


    except:
        return {'error':'invalid payload'}


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

    return render_template('Status.html', status= status, acc=acc, loss=loss, val_acc=val_acc, val_loss=val_loss)


if __name__ == '__main__':
    # Ponemos en marcha el servidor Flask
    app.run(host='0.0.0.0', port=port, debug=False)
