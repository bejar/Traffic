"""
.. module:: Misc

Misc
*************

:Description: Misc

    Miscelaneous functions

:Authors: bejar
    

:Version: 

:Created on: 15/02/2017 9:41 

"""

__author__ = 'bejar'


def load_config_file(nfile):
    '''
    Read the configuration from a json file

    :param nfile:
    :return:
    '''
    fp = open('./' + nfile + '.json', 'r')

    s = ''

    for l in fp:
        s += l

    return s


def transweights(weights):
    """
    Transforms class weights format from json to python
    :param weights:
    :return:
    """
    wtrans = {}
    for v in weights:
        wtrans[str(v)] = weights[v]
    return wtrans


def detransweights(weights):
    """
    Transforms class weights format from python to json
    :param weights:
    :return:
    """
    wtrans = {}
    for v in weights:
        wtrans[int(v)] = weights[v]
    return wtrans


def list_days_generator(year, month, iday, fday):
    """
    Generates a list of days

    :param year:
    :param month:
    :param iday:
    :param fday:
    :return:
    """
    ldays = []
    for v in range(iday, fday+1):
        ldays.append("%d%d%02d" % (year, month, v))
    return ldays


def name_days_file(ldays):
    """
    Generates a string with the dates

    :param ldays:
    :return:
    """
    return ldays[0] + '-' + ldays[-1]


def dist_time(time1, time2):
    """
    distance between two hours

    :param time1:
    :param time2:
    :return:
    """
    t1 = (time1 % 100) + (60 * ((time1 // 100) % 100))
    t2 = (time2 % 100) + (60 * ((time2 // 100) % 100))
    return t2 - t1
