from zipfile import ZipFile
import os
import Augmentor
import shutil
import subprocess
import ruamel.yaml
from box import Box

# get configuration file
cnf = Box.from_yaml(filename="../params.yaml", Loader=ruamel.yaml.Loader)

def fileToFolder(datadir, exten='jpeg'):
    '''
    for a given directory it will put every file in a directory inside this directory and name this
    newly created directory with the file name.
    example:
    # this is the input
      datadir/
          img1.jpeg
          img2.jpeg
    # this will be the new output
      datadir/
        img1/
           img1.jpeg
        img2/
           img2.jpeg

    datadir: directory that you want to transform, string
    exten: jpeg, png like extension of the file,
        other file types will be ignored if they are different than extension, string
    '''
    for image in os.listdir(datadir):
        new_folder_name, extension = image.split('.')
        if not extension == exten:
            continue
        new_folder_path = os.path.join(datadir, new_folder_name)
        if os.path.isdir(new_folder_path):
            shutil.rmtree(new_folder_path)
        os.mkdir(new_folder_path)
        os.rename(os.path.join(datadir, image), os.path.join(new_folder_path, image))
    print('successfully created folders inside %s' % datadir)


def augmentation_strategy(datadir, width=120, height=120, nsamples=50000):
    p = Augmentor.Pipeline(datadir)
    p.resize(probability=1,width=width,height=height,resample_filter=u'NEAREST')
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.zoom(probability=0.5, min_factor=1.1, max_factor=1.3)
    p.random_brightness(probability=0.3,min_factor=0.4,max_factor=1.4)
    p.random_color(probability=0.3,min_factor=0.5,max_factor=1.0)
    p.random_contrast(probability=0.3,min_factor=0.7,max_factor=1.0)
    #p.random_erasing()
    #p.random_distortion
    p.crop_random(probability=0.3,percentage_area=0.7)
    p.skew_tilt(probability=0.4,magnitude=0.5)
    p.skew_corner(probability=0.1,magnitude=0.3)
    p.sample(nsamples)

# todo: resizing image has to be added
# def movegroundtruth_to_train(datadir='train',width=120, height=120):
#     for classfolder in os.listdir(datadir):
#         if classfolder == 'output':
#             continue
#         for image in os.listdir(os.path.join(datadir, classfolder)):
#             input_path = os.path.join(datadir, classfolder, image)
#             output_path = os.path.join(datadir, 'output', classfolder, image)
#             shutil.move(input_path, output_path)


# apply strategies and organize folders
for newfolder in ['train', 'test']:
    fileToFolder(newfolder)
    augmentation_strategy(datadir=newfolder,
                          width=cnf.image.width,
                          height=cnf.image.height,
                          nsamples=cnf.augment.nsamples)
    #if newfolder == 'train':
        #movegroundtruth_to_train(datadir=newfolder)
    shutil.copytree(src=os.path.join(newfolder, 'output'), dst='output')
    shutil.rmtree(path=newfolder)
    shutil.move(src='output', dst=newfolder)

    # zip the folders but remove if they exist
    if os.path.isdir(newfolder + '.zip'):
        shutil.rmtree(newfolder + '.zip')
    command = ['zip', '-r', newfolder + '.zip', newfolder]
    subprocess.check_call(command)
    shutil.rmtree(newfolder)