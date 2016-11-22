"""
.. module:: CamTram

CamTram
*************

:Description: CamTram

    Tramos asociados a las camaras y el nombre del tramo, se guardan los dos tramos mas cercanos

:Authors: bejar
    

:Version: 

:Created on: 14/11/2016 14:05 

"""

__author__ = 'bejar'

from Util.Constants import data_path

class CamTram:

    ct = None
    def __init__(self):
        """

        """
        self.ct = {}
        f = open(data_path + 'CameraTram.txt', 'r')

        for line in f:
            cam, tram1, tram2, name1, name2 = line.split(',')

            self.ct[cam.strip()] = (int(tram1), int(tram2), name1.strip(), name2.strip())

        f.close()
