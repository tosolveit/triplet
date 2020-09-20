import os
import shutil
from zipfile import ZipFile
import tensorflow as tf


def extractzip(zippedfilename, outputfoldername):
    if os.path.isdir(outputfoldername):
        shutil.rmtree(outputfoldername)
    with ZipFile(zippedfilename, 'r') as zipObj:
       # Extract all the contents of zip file in different directory
       zipObj.extractall()
    os.rename(zippedfilename.split('.')[0], outputfoldername)


def check_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        print('available gpus:')
        for gpu in gpus:
            print(gpu)
    else:
        print('no gpus found on machine')