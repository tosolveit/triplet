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
    print('ilker')

    # Validate
    assert out == 'no gpus found on machine\n'

