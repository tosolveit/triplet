import os
import Augmentor
import shutil
import ruamel.yaml
from box import Box


def setup():
    # get configuration file path
    p = os.path.abspath(os.path.join(__file__, "../.."))
    params_path = os.path.join(p, 'params.yaml')

    # read config based on params, params.yaml is at the project root
    cnf = Box.from_yaml(filename=params_path, Loader=ruamel.yaml.Loader)

    # define train and test data path
    train_stage = os.path.join(p, 'data/train_stage')
    test_stage = os.path.join(p, 'data/test_stage')

    # source and destination paths to save processed outputs
    datasets = dict()
    datasets[train_stage] = os.path.join(p, 'data/train')
    datasets[test_stage] = os.path.join(p, 'data/test')
    return cnf, datasets


def file_to_folder(datadir, exten='jpeg'):
    """
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
    """
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


def augment_images(source_directory, output_directory, width=120, height=120, nsamples=50000):
    p = Augmentor.Pipeline(source_directory=source_directory,output_directory=output_directory)
    p.resize(probability=1, width=width, height=height, resample_filter=u'NEAREST')
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.zoom(probability=0.5, min_factor=1.1, max_factor=1.3)
    p.random_brightness(probability=0.3, min_factor=0.4, max_factor=1.4)
    p.random_color(probability=0.3, min_factor=0.5, max_factor=1.0)
    p.random_contrast(probability=0.3, min_factor=0.7, max_factor=1.0)
    p.crop_random(probability=0.3, percentage_area=0.7)
    p.skew_tilt(probability=0.4, magnitude=0.5)
    p.skew_corner(probability=0.1, magnitude=0.3)
    p.sample(nsamples)


def main():
    cnf, datasets = setup()
    for data_stage, data_dest in datasets.items():
        file_to_folder(data_stage)
        augment_images(source_directory=data_stage,
                       output_directory=data_dest,
                       width=cnf.image.width,
                       height=cnf.image.height,
                       nsamples=cnf.image.nsamples)


if __name__ == '__main__':
    main()

