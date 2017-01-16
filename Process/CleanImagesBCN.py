"""
.. module:: CleanImages

CleanImages
*************

:Description: CleanImages

    Detects images that are identical and only keeps the first one

:Authors: bejar
    

:Version: 

:Created on: 17/11/2016 12:34 

"""

import filecmp
import os
from Util.Cameras import Cameras
from Util.Constants import cameras_path
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

__author__ = 'bejar'

if __name__ == '__main__':
    day = '20170115'
    for cam in Cameras:

        ldir = glob.glob(cameras_path + day + '/*%s.gif'%cam)
        # print(ldir)

        lfiles = sorted(ldir)
        ldel = []
        for i in range(len(lfiles)-1):
            if filecmp.cmp(lfiles[i], lfiles[i+1], shallow=False):
                ldel.append(lfiles[i+1])
                print('R=', lfiles[i+1])

                # image1 = mpimg.imread(lfiles[i])
                # image2 = mpimg.imread(lfiles[i+1])
                # fig = plt.figure()
                # fig.set_figwidth(60)
                # fig.set_figheight(30)
                # sp1 = fig.add_subplot(1,2,1)
                # sp1.imshow(image1)
                # sp1 = fig.add_subplot(1,2,2)
                # sp1.imshow(image2)
                # plt.show()
                # plt.close()
        for f in ldel:
            print('Removing=', f)
            os.remove(f)

