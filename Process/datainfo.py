"""
.. module:: datainfo

datainfo
*************

:Description: datainfo

    Computes the itinerary that is closest to the camera

    This does not take into consideration the direction the camera is facing and that each itinerary has
    two directions. To select the correct itinerary both directions should be stored, this complicates
    how to select the data of status for the camera.

:Authors: bejar
    

:Version: 

:Created on: 07/11/2016 14:44 

"""

import numpy as np

from Util.Cameras import CamCoord, Cameras

__author__ = 'bejar'


def euc_dist(c1, c2):
    """
    Euclidean distance two coordinates

    :param c1:
    :param c2:
    :return:
    """
    return np.sqrt(((c1[0]-c2[0])**2)+((c1[1]-c2[1])**2))


path = '/home/bejar/Data/Traffic/'

tramos = {}

f = open(path + 'Tramos Barna.txt', 'r')

c = 0
for line in f:

    if c % 3 == 0:
        it = line.strip()
    if c % 3 == 1:
        val = line.strip()
    if c %3 == 2:
        tramos[it] = (val, line.strip())
    c += 1

# for v in tramos:
#     print(tramos[v])

f.close()

f = open(path + 'Itinerarios Barna.csv', 'r')

litin = []

for line in f:
    v = line.split(',')
    if v[1] in tramos:
        coord = tramos[v[1]][1].split(' ')
        lcoord = []
        #print(tramos[v[1]][0])
        for c in coord:
            vcoord = c.split(',')
            #print(vcoord)
            lcoord.append((float(vcoord[0]), float(vcoord[1])))
        litin.append([int(v[1]), tramos[v[1]][0], lcoord])

#print(len(litin))

# for v in sorted(litin):
#     print(v)


for cam in Cameras:
    coord = CamCoord[cam]
    mdist = 1
    mdist2 = 2
    res = [cam, None, None, None, None]
    for ni, street, iti in litin:

        for i in iti:
            if euc_dist(coord,i) < mdist:
                mdist2 = mdist
                mdist = euc_dist(coord,i)
                res[2] = res[1]
                res[1] = ni
                res[4] = res[3]
                res[3] = street
            elif euc_dist(coord,i) < mdist2:
                mdist2 = euc_dist(coord,i)
                res[2] = ni
                res[4] = street

    print('%s, %d, %d, %s, %s' % (res[0], res[1], res[2], res[3], res[4]))


