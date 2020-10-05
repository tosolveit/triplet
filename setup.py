import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import setup
from setuptools import find_packages


setup(
    name='triplet',
    version='0.0.1',
    description='Triplet loss model',
    author='Ilker Karapanca & Can Tosun',
    author_email='ilkerkarapanca@gmail.com',
    url='https://github.com/tosolveit/triplet',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    install_requires=['numpy>=1.18.5',
                      'annoy>=1.16.3',
                      'Pillow~=7.2.0',
                      'ruamel.yaml==0.16.12',
                      'python-box==5.1.1',
                      'tqdm==4.49.0',
                      'tensorflow==2.3.1',
                      'tensorflow-addons>=0.11.2',
                      'Augmentor>=0.2.8'
                      ],
    extras_require={
        'dev': ['pytest==6.1.0','pytest-mock==3.3.1'],
    }
)
