"""
.. module:: DataTram

DataTram
*************

:Description: DataTram

    Recupera la informacion del estado del transito del fichero de tramos

:Authors: bejar
    

:Version: 

:Created on: 14/11/2016 14:13 

"""

__author__ = 'bejar'


from Constants import data_path


class DataTram:

    dt = None
    date = None

    def __init__(self, fname):
        """

        """
        f = open(fname, 'r')
        self.dt = {}
        for line in f:
            tram, date, cl, pcl = line.split('#')
            self.dt[int(tram)] = (int(cl), int(pcl))
        self.date = int(date[0:-2])

        f.close()
