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
    gpus = len(tf.config.list_physical_devices('GPU'))
    if gpus > 0:
        print(f'available gpus:{gpus}')
    else:
        print('no gpus found on machine')