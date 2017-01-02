"""
.. module:: GetImages

GetImages
*************

:Description: GetImages

    

:Authors: bejar
    

:Version: 

:Created on: 27/10/2016 8:23 

"""

import os.path
import time

import requests

from Util.Cameras import Cameras
from Util.Constants import cameras_path, data_path, status_path
from Util.Webservice import inform_webservice

__author__ = 'bejar'

state = 0

while True:
    todaypath = time.strftime('%Y%m%d', time.localtime(int(time.time())-600))
    if not os.path.exists(cameras_path + todaypath):
        os.mkdir(cameras_path + todaypath)
        os.mkdir(status_path + todaypath)

    rtime = str((int(time.time())-600)*1000)
    ptime = time.strftime('%Y%m%d%H%M', time.localtime(int(time.time())-600))

    try:
        print('%s Retrieving Traffic Status' % time.strftime('%H:%M %d-%m-%Y',time.localtime()))
        req = requests.get('http://www.bcn.cat/transit/dades/dadestrams.dat')
        if req.status_code == 200:
            tram = req.content
            with open(status_path + todaypath + '/' + '%s-dadestram.data' % (ptime), 'wb') as handler:
                    handler.write(tram)

            tram = requests.get('http://www.bcn.cat/transit/dades/dadesitineraris.dat').content
            with open(status_path + todaypath + '/' + '%s-dadesitineraris.data' % (ptime), 'wb') as handler:
                    handler.write(tram)

            #if (state % 3) == 0:
            print('%s Retrieving Cameras' % time.strftime('%H:%M %d-%m-%Y',time.localtime()))
            for cam in Cameras:
                img_data = requests.get('http://www.bcn.cat/transit/imatges/%s.gif?a=1&time=%s' % (cam,rtime)).content
                with open(cameras_path + todaypath + '/' +'%s-%s.gif' % (ptime, cam), 'wb') as handler:
                    handler.write(img_data)

            inform_webservice('BCN', 0)
            # state += 1
            # if state > 1002:
            #     state = 0
        else:
            print('Service not available')
            inform_webservice('BCN', 2)
    except Exception:
        pass

    time.sleep(15 * 60)



