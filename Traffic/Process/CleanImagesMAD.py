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
from Utilities.Constants import cameras_path_MAD, data_path_MAD
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import csv

__author__ = 'bejar'

if __name__ == '__main__':
    day = '20170213'

    lcameras = []
    with open(data_path_MAD + 'MAD_cameras.txt', 'r') as csvfile:
        camreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in camreader:
            lcameras.append(row[0])

    for cam in lcameras:

        print(cam)
        ldir = glob.glob(cameras_path_MAD + day + '/*%s.jpg'%cam)

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

