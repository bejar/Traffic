"""
.. module:: GetImagesMadrid

GetImagesMadrid
*************

:Description: GetImagesMadrid

    

:Authors: bejar
    

:Version: 

:Created on: 17/11/2016 8:54 

"""

import csv
import os
import time

import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError
import urllib3
from bs4 import BeautifulSoup
from joblib import Parallel, delayed

from Util.Constants import data_path_MAD, cameras_path_MAD, status_path_MAD
from Util.Webservice import inform_webservice

__author__ = 'bejar'


# Niveles de Servicio http://informo.munimadrid.es/informo/tmadrid/tramos.kml
# Camaras http://informo.munimadrid.es/informo/tmadrid/cctv.kml
# intensidades http://informo.munimadrid.es/informo/tmadrid/intensidades.kml


def get_info_cameras():
    """
    Retrieves the file with the Madrid traffic cameras and saves it in a file
    :return:
    """
    url = 'http://informo.munimadrid.es/informo/tmadrid/cctv.kml'
    http = urllib3.PoolManager()

    try:
        data = http.request('GET', url)

    except urllib3.HTTPError:
        pass

    # print(data.data)
    soup = BeautifulSoup(str(data.data), "lxml")
    lcameras = []
    for camera, coord, name in zip(soup.find_all('description'), soup.find_all('coordinates'), soup.find_all(attrs={"name": "Nombre"})):
        url_camera = str(camera)
        url_camera = url_camera[url_camera.find('http'):url_camera.find('?')]
        vcoord = str(coord)
        long = vcoord.split(',')[0].strip()
        long = long[long.find('>')+1:]
        lat = vcoord.split(',')[1].strip()
        nombre = str(name)
        nombre = nombre[nombre.find('ue>')+3:nombre.find('</')]
        # print(nombre, url_camera, long,lat)
        lcameras.append((nombre, url_camera, long,lat))

    # f = open(data_path_MAD + 'MAD_cameras.txt', 'w')

    with open(data_path_MAD + 'MAD_cameras.txt', 'w') as csvfile:
        camwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for camera in lcameras:
            camwriter.writerow(camera)

def retrieve_camera(cam, name, ptime):
    """
    Retrieves the image of a camera

    :return:
    """
    try:
        resp = requests.get(cam)
        if resp.status_code != 200:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        img_data = resp.content

        with open(cameras_path_MAD + todaypath + '/' + '%s-%s.jpg' % (ptime, name), 'wb') as handler:
            handler.write(img_data)
    except ConnectionError:
        pass
    except ChunkedEncodingError:
        pass

if __name__ == '__main__':
    # get_info_cameras()

    niveles = 'http://informo.munimadrid.es/informo/tmadrid/tramos.kml'
    intensidades = 'http://informo.munimadrid.es/informo/tmadrid/intensidades.kml'

    lcameras = []
    lnames = []
    with open(data_path_MAD + 'MAD_cameras.txt', 'r') as csvfile:
        camreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in camreader:
            lcameras.append(row[1])
            lnames.append(row[0])

    while True:
        todaypath = time.strftime('%Y%m%d', time.localtime(int(time.time())))
        if not os.path.exists(cameras_path_MAD + todaypath):
            os.mkdir(cameras_path_MAD + todaypath)
            os.mkdir(status_path_MAD + todaypath)

        ptime = time.strftime('%Y%m%d%H%M', time.localtime(int(time.time())))

        print('%s Retrieving Traffic Status' % time.strftime('%H:%M %d-%m-%Y',time.localtime()))

        try:
            tram = requests.get(niveles).content
            with open(status_path_MAD + todaypath + '/' + '%s-niveles.kml' % ptime, 'wb') as handler:
                    handler.write(tram)
        except ChunkedEncodingError:
            pass
        except ConnectionError:
            pass

        try:
            tram = requests.get(intensidades).content
            with open(status_path_MAD + todaypath + '/' + '%s-intensidades.kml' % ptime, 'wb') as handler:
                    handler.write(tram)
        except ChunkedEncodingError:
            pass
        except ConnectionError:
            pass

        print('%s Retrieving Cameras' % time.strftime('%H:%M %d-%m-%Y',time.localtime()))

        Parallel(n_jobs=-1)(
            delayed(retrieve_camera)(cam, name, ptime) for cam, name in zip(lcameras,lnames))

        inform_webservice('MAD', 0)
        time.sleep(3 * 60)
