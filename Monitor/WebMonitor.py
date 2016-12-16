__author__ = 'bejar'

import socket
import time

from pymongo import MongoClient
from Config.Private import mongodata
from flask import Flask, request, render_template
#from Parameters.Private import WS_port


# Configuration stuff
hostname = socket.gethostname()
port = 80

app = Flask(__name__)




@app.route('/Status')
def info():
    """
    Status de las ciudades
    """

    client = MongoClient(mongodata.server)
    db = client[mongodata.db]
    col = db[mongodata.col]

    vals = col.find({'done':False}, {'acc':1, 'loss': 1, 'val_acc':1, 'val_loss':1})



    return render_template('Status.html')


if __name__ == '__main__':
    # Ponemos en marcha el servidor Flask
    app.run(host='0.0.0.0', port=port, debug=False)
