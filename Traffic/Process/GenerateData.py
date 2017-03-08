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
    parser.add_argument('--test', action='store_true', default=False, help='Data generated for test')
    parser.add_argument('--imgord', default='th', help='Image Ordering')
    parser.add_argument('--delay', default='15', help='Time Delay')
    parser.add_argument('--idate', default='20161101', help='First day')
    parser.add_argument('--fdate', default='20161130', help='Final day')
    # Dicards images from an hour larger or equal than the first element and  less or equal than the second element
    parser.add_argument('--hourban', nargs='+', default=[None, None], help='Initial ban hour', type=int)

    parser.add_argument('--augmentation', nargs='+', default=[], help='Use data augmentation for certain classes', type=int)
    # The badpixels filted discards the images using a histogram of colors, it has two values
    # number of bins for the histogram and a percentage that is the threshold for discarding the image
    parser.add_argument('--badpixels', nargs='+', default=None, help='Apply the badpixels filter to the images', type=int)
    # for compresssing uses as value in the parameter gzip
    parser.add_argument('--compress', action='store_true', default=False, help='Compression for the HDF5 file')

    args = parser.parse_args()

    z_factor = float(args.zoom)
    imgord = args.imgord
    chunk = int(args.chunk)
    mxdelay = int(args.delay)
    days = list_range_days_generator(args.idate, args.fdate)
    compress = 'gzip' if args.compress else None

    if len(args.hourban) > 0:
        if args.hourban[0] is None or 0<= args.hourban[0] <= 23:
            ihour = args.hourban[0]
        else:
            raise Exception('Parameters for HOURBAN incorrect hours in [0,23]')

    if len(args.hourban) == 2:
        if args.hourban[0] is None or 0<= args.hourban[1] <= 23:
            fhour = args.hourban[1]
        else:
            raise Exception('Parameters for HOURBAN incorrect hours in [0,23]')
    else:
        fhour = None


    if args.badpixels is not None:
        if len(args.badpixels) == 2:
            nbin = args.badpixels[0]
            if 0 <= args.badpixels[1] <= 100:
                perc = args.badpixels[1]
            else:
                raise Exception('Parameters for BADPIXELS incorrect Percentage in [0,100]')
        else:
            raise Exception('Parameters for BADPIXELS incorrect')
    else:
        nbin, perc = None, None

    print 'Generating data for:'
    print 'ID = ', days[0]
    print 'FD = ', days[-1]
    print 'ZF = ', z_factor
    print 'IO = ', imgord
    print 'CHS = ', chunk
    print 'DLY = ', mxdelay
    print 'TEST = ', args.test
    if ihour is not None:
        print 'IHBAN = ', args.hourban[0]
    if fhour is not None:
        print 'FHBAN = ', args.hourban[1]
    if args.badpixels is not [None, None]:
        print 'BPXS = ', args.badpixels
    print 'AUG = ', args.augmentation
    print 'COMPRESS = ', compress

    print
    print 'Generating days ...'
    for day in days:
        print 'Generating %s' % day
        generate_labeled_dataset_day(process_path, day, z_factor, mxdelay=mxdelay, onlyfuture=False, imgordering=imgord,
                                     augmentation=args.augmentation, hourban=(ihour, fhour), badpixels=(nbin,perc))

    print
    print 'Days info ...'
    info_dataset(process_path, days, z_factor, imgordering=imgord)

    print
    print 'Generating HDF5 file ...'
    generate_training_dataset(process_path, days, chunk=chunk, z_factor=z_factor, imgordering=imgord, test=args.test,
                              compress=compress)
