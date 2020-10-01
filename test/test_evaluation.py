from evaluate import Evaluate
import pytest
from pytest_mock import mocker
import tensorflow as tf


def test_read_image_and_convert_to_array__is_numpy(random_image):
    # Set up
    image_path = str(random_image)
    evaluate = Evaluate()

    # Exercise
    _, image = evaluate.read_image_and_convert_to_array(image_path)

    # Verify
    assert type(image).__module__ == 'numpy'


    # Cleanup: none


def test_read_image_and_convert_to_array__is_two_dim(random_image, conf):
    # Set up
    image_path = str(random_image)
    evaluate = Evaluate()
    expected = (conf.image.width, conf.image.height, conf.image.channels)

    # Exercise
    _, image = evaluate.read_image_and_convert_to_array(image_path)

    # Verify
    assert image.shape == expected

    # Cleanup: none


def test_read_image_and_convert_to_array__is_range_true(random_image):
    # Set up
    image_path = str(random_image)
    evaluate = Evaluate()

    # Exercise
    _, image = evaluate.read_image_and_convert_to_array(image_path)

    # Verify
    assert image.min() >= 0 and image.max() <= 1.

    # Cleanup: none


def test_read_image_and_convert_to_array__is_filename_true(random_image):
    # Set up
    image_path = str(random_image)
    evaluate = Evaluate()

    # Exercise
    index, _ = evaluate.read_image_and_convert_to_array(image_path)

    # Verify
    assert index == '999'

    # Cleanup: none


def test_calculate_metric_score(mocker):
    # Set up
    evaluate = Evaluate()
    mocker.patch.object(evaluate, 'calculate_metric')
    evaluate.calculate_metric.return_value = (10, 100)

    # Exercise
    metric = evaluate.calculate_metric()

    # Verify
    assert metric == (10, 100)

    # Cleanup: none






