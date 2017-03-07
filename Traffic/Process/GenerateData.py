"""
.. module:: GenerateData

GenerateData
*************

:Description: GenerateData

    

:Authors: bejar
    

:Version: 

:Created on: 20/02/2017 8:26 

"""
import argparse
from Traffic.Process.DatasetProcess import generate_labeled_dataset_day, generate_training_dataset, info_dataset
from Traffic.Util.Misc import list_range_days_generator
from Traffic.Config.Constants import process_path

__author__ = 'bejar'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--zoom', default='0.25', help='Zoom Factor')
    parser.add_argument('--chunk', default='1024', help='Chunk size')
    # --test flag for including all data in the HDF5 file (last chunk is not the same size than the rest)
    parser.add_argument('--test', default=False, help='Data generated for test', type=bool)
    parser.add_argument('--imgord', default='th', help='Image Ordering')
    parser.add_argument('--delay', default='15', help='Time Delay')
    parser.add_argument('--idate', default='20161101', help='First day')
    parser.add_argument('--fdate', default='20161130', help='Final day')
    parser.add_argument('--augmentation', default=False, help='Use data augmentation')
    # for compresssing uses as value in the parameter gzip
    parser.add_argument('--compress', default=None, help='Compression for the HDF5 file')

    args = parser.parse_args()

    z_factor = float(args.zoom)
    imgord = args.imgord
    chunk = int(args.chunk)
    mxdelay = int(args.delay)
    days = list_range_days_generator(args.idate, args.fdate)

    print 'Generating data for:'
    print 'ID = ', days[0]
    print 'FD = ', days[-1]
    print 'ZF = ', z_factor
    print 'IO = ', imgord
    print 'CHS = ', chunk
    print 'DLY = ', mxdelay
    print 'TEST = ', args.test
    print 'COMPRESS = ', args.compress

    print
    print 'Generating days ...'
    for day in days:
        print 'Generating %s' % day
        generate_labeled_dataset_day(process_path, day, z_factor, mxdelay=mxdelay, onlyfuture=False, imgordering=imgord,
                                     augmentation=args.augmentation)

    print
    print 'Days info ...'
    info_dataset(process_path, days, z_factor, imgordering=imgord)

    print
    print 'Generating HDF5 file ...'
    generate_training_dataset(process_path, days, chunk=chunk, z_factor=z_factor, imgordering=imgord, test=args.test,
                              compress=args.compress)
