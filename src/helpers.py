import os
import shutil
from zipfile import ZipFile

def extractzip(zippedfilename, outputfoldername):
    if os.path.isdir(outputfoldername):
        shutil.rmtree(outputfoldername)
    with ZipFile(zippedfilename, 'r') as zipObj:
       # Extract all the contents of zip file in different directory
       zipObj.extractall()
    os.rename(zippedfilename.split('.')[0], outputfoldername)