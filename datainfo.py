"""
.. module:: datainfo

datainfo
*************

:Description: datainfo

    

:Authors: bejar
    

:Version: 

:Created on: 07/11/2016 14:44 

"""

from Cameras import CamCoord, Cameras
import numpy as np
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
    res = None
    for ni, street, iti in litin:
        for i in iti:
            if euc_dist(coord,i) < mdist:
                mdist = euc_dist(coord,i)
                res = (cam, ni, street)
    print(res[0], ',', res[1], ',', res[2])


