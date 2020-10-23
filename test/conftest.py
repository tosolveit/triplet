import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from box import Box
import ruamel.yaml
from annoy import AnnoyIndex
from evaluate import  Evaluate
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Lambda, Input)


@pytest.fixture(scope='session')
def random_image(tmpdir_factory):
    a = np.random.rand(120, 120, 3) * 255
    im_out = Image.fromarray(a.astype('uint8')).convert('RGB')
    folder_path = tmpdir_factory.mktemp("data")
    image_path = folder_path.join('%d.jpeg' % 999)
    im_out.save(str(image_path))
    return image_path


@pytest.fixture
def random_images(tmpdir_factory):
    a = np.random.rand(120, 120, 3) * 255
    im_out = Image.fromarray(a.astype('uint8')).convert('RGB')
    folder_path = tmpdir_factory.mktemp("datas")
    for i in range(2):
        image_path = folder_path.join('%d.jpeg' % i)
        im_out.save(str(image_path))
    return folder_path


@pytest.fixture(scope='session')
def conf():
    params_file_name = 'params.yaml'
    project_root = (Path(__file__) / ".." / "..")
    params_path = project_root.joinpath(params_file_name).resolve()
    cnf = Box.from_yaml(filename=params_path, Loader=ruamel.yaml.Loader)
    return cnf


@pytest.fixture(scope='session')
def mock_model_saved(tmpdir_factory):
    # create model that returns the input only
    input = Input((120, 120, 3))
    output = Lambda(lambda x: x)(input)
    model = Model(input, output)

    # save mock model
    fn = tmpdir_factory.mktemp("mock-model").join("model-rijk.h5")
    model.save(str(fn))
    return fn


@pytest.fixture(scope='session')
def mock_model():
    # create model that returns the input only
    input = Input((120, 120, 3))
    output = Lambda(lambda x: x)(input)
    model = Model(input, output)
    return model


@pytest.fixture(scope='session')
def mock_create_index():
    f = 128
    arr = np.random.random(f)
    t = AnnoyIndex(f, 'euclidean')
    t.add_item(1, arr)  # adding an item to index
    return t
