from evaluate import Evaluate
import pytest
from pytest_mock import mocker
import tensorflow as tf
import os


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
    evaluate.calculate_metric.return_value = 0.89

    # Exercise
    metric = evaluate.calculate_metric()

    # Verify
    assert metric == 0.89

    # Cleanup: none


def test_write_metric(mocker, tmpdir):
    # Set up

    # mock the calculate_metric and set it to metric variable
    evaluate = Evaluate()
    mocker.patch.object(evaluate, 'calculate_metric')
    evaluate.calculate_metric.return_value = 0.89
    metric = evaluate.calculate_metric()

    expected = "precision @1: " + str(metric) + "\n"
    metrics_file = os.path.join(tmpdir, 'metrics_file.txt')

    # Exercise

    # will call calculate_metric internally
    evaluate.write_metric(metrics_file=metrics_file)

    # Verify
    with open(metrics_file, 'r') as f:
        actual = f.read()
    assert actual == expected


def test_sample_from_directory(mocker):
    # Set up
    evaluate = Evaluate(sample_size=2)
    mocker.patch.object(Evaluate, '_get_list_of_the_files_in_a_directory')
    Evaluate._get_list_of_the_files_in_a_directory.return_value = [1, 3, 5, 6]
    expected = Evaluate._get_list_of_the_files_in_a_directory('a_mock_directory')

    # Exercise
    samples = evaluate.sample_from_directory('a_mock_directory')

    # Verify
    assert len(samples) == 2
    assert set(samples).issubset(expected).__eq__(True)









