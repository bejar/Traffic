"""
.. module:: Generate_Dataset

Generate_Dataset
*************

:Description: Generate_Dataset

    Generate a dataset for traffic images

:Authors: bejar
    

:Version: 

:Created on: 14/11/2016 12:53 

"""

import glob

import numpy as np

from Process.CamTram import CamTram
from Util.Constants import cameras_path, status_path
from Util.DataTram import DataTram

__author__ = 'bejar'


def generate_classification_dataset(day):
    """
    Generates a dictionary with the dates of the images with lists that contain the camera name and current and predicted
    traffic status

    :param day:
    :return:
    """

    #day = '20161031'

    CTram = CamTram()

    ldir = glob.glob(cameras_path + day + '/*.gif')

    camdic = {}

    for f in sorted(ldir):
        name = f.split('.')[0].split('/')[-1]
        time, place = name.split('-')
        # print(time, place, CTram.ct[place])
        if int(time) in camdic:
            camdic[int(time)].append(place)
        else:
            camdic[int(time)] = [place]

    # print(camdic)

    ldir = glob.glob(status_path + day + '/*-dadestram.data')
    ldata = []
    for f in sorted(ldir):
        ldata.append(DataTram(f))


    assoc = {}

    for imgtime in sorted(camdic):
        # Look for the status and forecast closer to the image but always in the future
        dmin = None
        dmin2 = None
        vmin = 10000
        for d in ldata:
            if vmin > np.abs(imgtime - d.date): # Only if it is ahead in time
                if imgtime - d.date >0:
                    vmin = np.abs(imgtime - d.date)
                    dmin = d
                    dmin2 = dmin
        if dmin is not None and dmin2 is not None:
            lclass = []
            for img in camdic[imgtime]:
                tram = CTram.ct[img][0]
                # print(imgtime, dmin.dt[tram], img)
                # store for an image of that time the name, closest status, prediction and next status
                lclass.append((img, dmin.dt[tram][0], dmin.dt[tram][1], dmin2.dt[tram][0]))
            assoc[imgtime] = lclass

    return assoc
