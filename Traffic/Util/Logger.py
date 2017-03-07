"""
.. module:: Logger

Logger
******

:Description: Logger

    Different Auxiliary functions used for different purposes

:Authors:
    bejar

:Version: 

:Date:  18/11/2016
"""

import logging

from Traffic.Config.Constants import results_path

__author__ = 'bejar'


def config_logger(silent=False, filename=None, logpath=results_path):
    if filename is not None:
        logging.basicConfig(filename=logpath + '/' + filename + '.log', filemode='w')

    # Logging configuration
    logger = logging.getLogger('log')
    if silent:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    if silent:
        console.setLevel(logging.ERROR)
    else:
        console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('log').addHandler(console)
    return logger
