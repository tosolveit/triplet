import os
import shutil
from zipfile import ZipFile
import tensorflow as tf


def unzip(fname,oname=None):
    with ZipFile(fname, 'r') as zipObj:
        if oname is not None:
            zipObj.extractall(oname)
        else:
            zipObj.extractall()
        print('extracted to: %s' % fname)


def extractzip(zippedfilename, outputfoldername=None):
    if outputfoldername is not None:
        if os.path.isdir(outputfoldername):
            shutil.rmtree(outputfoldername)
            print('remove already existing folder: %s' % outputfoldername)
        unzip(zippedfilename, outputfoldername)
    else:
        unzip(zippedfilename)


def check_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        print('available gpus:')
        for gpu in gpus:
            print(gpu)
    else:
        print('no gpus found on machine')