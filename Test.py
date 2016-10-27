"""
.. module:: Test

Test
*************

:Description: Test

    

:Authors: bejar
    

:Version: 

:Created on: 27/10/2016 14:44 

"""

__author__ = 'bejar'


import requests
from Cameras import Cameras
import time

__author__ = 'bejar'



path = '/home/bejar/storage/Data/Traffic/Test/'

while True:
    print('%s Retrieving Cameras' % time.strftime('%H:%M %d-%m-%Y',time.localtime()))
    print(time.ctime(int(time.time())-600))
    rtime = str((int(time.time())-600)*1000)
    ctime = time.strftime('%Y%m%d%H%M', time.localtime())

    for cam in Cameras:
        #print('Retrieving %s ...' % cam)

        img_data = requests.get('http://www.bcn.cat/transit/imatges/%s.gif?a=1&time=%s' % (cam,rtime)).content


        with open(path+'%s-%s.gif' % (ctime, cam), 'wb') as handler:
            handler.write(img_data)
        time.sleep(10)

    time.sleep(15 * 60)
