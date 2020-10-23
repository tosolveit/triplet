from helpers import check_gpus
import tensorflow as tf


def test_unzip():
    pass


def test_extractzip():
    pass


def test_check_gpus_empty(mocker, capsys):
    # Set up
    mocker.patch.object(tf.config,
                        'list_physical_devices',)
    tf.config.list_logical_devices.return_value = []

    # Execute
    check_gpus()
    out, err = capsys.readouterr()

    # Validate
    assert out == 'no gpus found on machine\n'


def test_check_gpus_multi(mocker, capsys):
    # Set up
    mocker.patch('tensorflow.config.list_physical_devices', return_value=[1, 2])

    # Execute
    check_gpus()
    out, _ = capsys.readouterr()

    # Validate
    assert out == 'available gpus:2\n'





