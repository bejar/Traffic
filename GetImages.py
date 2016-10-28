"""
.. module:: GetImages

GetImages
*************

:Description: GetImages

    

:Authors: bejar
    

:Version: 

:Created on: 27/10/2016 8:23 

"""


import requests
from Cameras import Cameras
import time

__author__ = 'bejar'


path = '/home/bejar/storage/Data/Traffic/'

camret = 0
while True:
    rtime = str((int(time.time())-600)*1000)
    ptime = int(time.time())-600

    print('%s Retrieving Traffic Status' % time.strftime('%H:%M %d-%m-%Y',time.localtime()))

    tram = requests.get('http://www.bcn.cat/transit/dades/dadestrams.dat').content
    with open(path +'Status/' + '%s-dadestram.data' % (ptime), 'wb') as handler:
            handler.write(tram)

    tram = requests.get('http://www.bcn.cat/transit/dades/dadesitineraris.dat').content
    with open(path + 'Status/' + '%s-dadesitineraris.data' % (ptime), 'wb') as handler:
            handler.write(tram)

    if camret == 0:
        print('%s Retrieving Cameras' % time.strftime('%H:%M %d-%m-%Y',time.localtime()))
        for cam in Cameras:
            img_data = requests.get('http://www.bcn.cat/transit/imatges/%s.gif?a=1&time=%s' % (cam,rtime)).content
            with open(path + 'Cameras/' +'%s-%s.gif' % (ptime, cam), 'wb') as handler:
                handler.write(img_data)

    time.sleep(5 * 60)
    camret += 1
    if camret == 3:
        camret = 0


