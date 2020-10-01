import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from box import Box
import ruamel.yaml
from evaluate import  Evaluate


@pytest.fixture(scope='session')
def random_image(tmpdir_factory):
    a = np.random.rand(120, 120, 3) * 255
    im_out = Image.fromarray(a.astype('uint8')).convert('RGB')
    folder_path = tmpdir_factory.mktemp("data")
    image_path = folder_path.join('%d.jpeg' % 999)
    im_out.save(str(image_path))
    return image_path


@pytest.fixture(scope='session')
def conf():
    params_file_name = 'params.yaml'
    project_root = (Path(__file__) / ".." / "..")
    params_path = project_root.joinpath(params_file_name).resolve()
    cnf = Box.from_yaml(filename=params_path, Loader=ruamel.yaml.Loader)
    return cnf


@pytest.fixture(autouse=True)
def patch_calculate_metric(monkeypatch):
    def substitute_calculate_metric():
        score, total = 900, 1000
        return score, total
    evaluate = Evaluate()
    monkeypatch.setattr(evaluate, 'calculate_metric', substitute_calculate_metric)
