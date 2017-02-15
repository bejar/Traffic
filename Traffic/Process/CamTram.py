"""
.. module:: CamTram

CamTram
*************

:Description: CamTram

    Street segment associated to the cameras and name of the segment, the two closest segments are stored

:Authors: bejar
    

:Version: 

:Created on: 14/11/2016 14:05 

"""

__author__ = 'bejar'


class CamTram:

    ct = None

    def __init__(self):
        """

        """
        self.ct = {}
        f = open('./CameraTram.txt', 'r')

        for line in f:
            cam, tram1, tram2, name1, name2 = line.split(',')

            self.ct[cam.strip()] = (int(tram1), int(tram2), name1.strip(), name2.strip())

        f.close()
