"""
.. module:: Webservice

Webservice
*************

:Description: Webservice

    

:Authors: bejar
    

:Version: 

:Created on: 18/01/2017 7:36 

"""

__author__ = 'bejar'


import requests

__author__ = 'bejar'

WS_port = 8870
Webservice = "http://polaris.lsi.upc.edu:8870/Update"

def inform_webservice(city, status):
    """
    Sends status report to webservice
    :return:
    """
    requests.get(Webservice, params={'content': 'traffic-' + city, 'count': status, 'delta': status})
