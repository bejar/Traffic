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
from Traffic.Config.Cameras import Cameras
from Traffic.Config.Constants import cameras_path
import glob
import argparse

__author__ = 'bejar'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--day', default=None, help='Day to clean')
    args = parser.parse_args()

    # day = '20170216'
    day = args.day
    for cam in Cameras:

        ldir = glob.glob(cameras_path + day + '/*%s.gif' % cam)

        lfiles = sorted(ldir)
        ldel = []
        for i in range(len(lfiles) - 1):
            if filecmp.cmp(lfiles[i], lfiles[i + 1], shallow=False):
                ldel.append(lfiles[i + 1])
                print('R=', lfiles[i + 1])
        for f in ldel:
            print('Removing=', f)
            os.remove(f)
