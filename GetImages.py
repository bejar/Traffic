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
import os.path

__author__ = 'bejar'


path = '/home/bejar/storage/Data/Traffic/'

while True:
    todaypath = time.strftime('%Y%m%d', time.localtime(int(time.time())-600))
    if not os.path.exists(path + 'Cameras/' + todaypath):
        os.mkdir(path + 'Cameras/' + todaypath)
        os.mkdir(path + 'Status/' + todaypath)

    rtime = str((int(time.time())-600)*1000)
    ptime = time.strftime('%Y%m%d%H%M', time.localtime(int(time.time())-600))

    print('%s Retrieving Traffic Status' % time.strftime('%H:%M %d-%m-%Y',time.localtime()))

    tram = requests.get('http://www.bcn.cat/transit/dades/dadestrams.dat').content
    with open(path +'Status/'  + todaypath + '/' + '%s-dadestram.data' % (ptime), 'wb') as handler:
            handler.write(tram)

    tram = requests.get('http://www.bcn.cat/transit/dades/dadesitineraris.dat').content
    with open(path + 'Status/' + todaypath + '/' + '%s-dadesitineraris.data' % (ptime), 'wb') as handler:
            handler.write(tram)

    print('%s Retrieving Cameras' % time.strftime('%H:%M %d-%m-%Y',time.localtime()))
    for cam in Cameras:
        img_data = requests.get('http://www.bcn.cat/transit/imatges/%s.gif?a=1&time=%s' % (cam,rtime)).content
        with open(path + 'Cameras/'  + todaypath + '/' +'%s-%s.gif' % (ptime, cam), 'wb') as handler:
            handler.write(img_data)

    time.sleep(15 * 60)



