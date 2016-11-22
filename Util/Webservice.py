"""
.. module:: Webservice

Webservice
******

:Description: Webservice

    Different Auxiliary functions used for different purposes

:Authors:
    bejar

:Version: 

:Date:  18/11/2016
"""

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