from generate_augmentation import file_to_folder
from generate_augmentation import augment_images
import pytest
from pytest_mock import mocker
import os
from numpy.testing import assert_array_equal
import numpy as np


def test_file_to_folder(random_images):
    # Setup

    # Exercise
    file_to_folder(random_images)
    actual = np.array(os.listdir(str(random_images)))
    actual.sort()

    # Verify
    expected = ['0', '1']
    assert_array_equal(actual, expected)


def test_augment_images(random_images, tmpdir):
    #  Setup

    #  Exercise

    augment_images(source_directory=random_images,
                   output_directory=tmpdir,
                   width=120,
                   height=120,
                   nsamples=10
                   )

    # Verify
    assert len(os.listdir(tmpdir)) == 10
    assert os.listdir(tmpdir)[0].split('.')[-1] == 'jpeg'


def test_main():
    pass
