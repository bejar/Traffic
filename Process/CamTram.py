"""
.. module:: CamTram

CamTram
*************

:Description: CamTram

    Tramos asociados a las camaras y el nombre del tramo

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
        f = open(data_path+ 'CameraTram.txt', 'r')

        for line in f:
            cam, tram, name = line.split(',')

            self.ct[cam.strip()] = (int(tram), name.strip())

        f.close()
