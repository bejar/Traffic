"""
.. module:: datainfo

datainfo
*************

:Description: datainfo

    

:Authors: bejar
    

:Version: 

:Created on: 07/11/2016 14:44 

"""

__author__ = 'bejar'


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
        print(tramos[v[1]][0])
        for c in coord:
            vcoord = c.split(',')
            print(vcoord)
            lcoord.append((vcoord[0], vcoord[1]))
        litin.append([int(v[1]), tramos[v[1]][0], lcoord])

print(len(litin))

for v in sorted(litin):
    print(v)