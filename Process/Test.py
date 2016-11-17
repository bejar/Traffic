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

import time

import requests

__author__ = 'bejar'



path = '/home/bejar/storage/Data/Traffic/Test/'

itime = int(time.time()-600)


for i in range(20):

    rtime = str(((itime-(i*900))*1000))
    ctime = time.strftime('%Y%m%d%H%M', time.localtime())

    print(time.ctime(itime-(i*900)))

    img_data = requests.get('http://www.bcn.cat/transit/imatges/TunelRovira.gif?a=1&time=%s' % rtime).content


    with open(path+'%s-TunelRovira.gif' % str(itime-(i*900)), 'wb') as handler:
        handler.write(img_data)
        time.sleep(2)
