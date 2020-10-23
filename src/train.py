import tensorflow as tf
import tensorflow_addons as tfa
import ruamel.yaml
from box import Box
from datetime import datetime

from processinput import ProcessInput
from gempoolinglayer import GeMPoolingLayer
from helpers import check_gpus
import os


def setup():
    # check if gpus are found on the machine and list them
    check_gpus()

    # get configuration file path
    p = os.path.abspath(os.path.join(__file__, "../.."))
    params_path = os.path.join(p, 'params.yaml')

    # read config based on params, params.yaml is at the project root
    cnf = Box.from_yaml(filename=params_path, Loader=ruamel.yaml.Loader)

    # define input data
    traindata = os.path.join(p, 'data/train')

    # create and model path
    modelpath = os.path.join(p, cnf.model.name)

    # create tensorflow dataset object
    processinput = ProcessInput(data_dir=traindata,
                                num_classes_per_batch=cnf.triplet.num_classes_per_batch,
                                num_images_per_class=cnf.triplet.num_images_per_class,
                                channels=cnf.image.channels,
                                img_width=cnf.image.width,
                                img_height=cnf.image.height,
                                file_extension=cnf.image.extension)
    dataset = processinput.train_input_fn()

    reg = tf.keras.regularizers
    input_shape = (cnf.image.height, cnf.image.width, cnf.image.channels)
    return input_shape, cnf, p, dataset, modelpath


def train(input_shape, cnf, p, dataset, modelpath):

    imagenet = tf.keras.applications.Xception(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False)

    imagenet.trainable = False

    # embedding model
    x_input = tf.keras.Input(shape=input_shape)
    x = imagenet(x_input)
    x = GeMPoolingLayer()(x)
    x = tf.keras.layers.Dense(cnf.model.embeddingsize, activation='softplus', kernel_regularizer=reg.l2(), dtype='float32')(x)

    model = tf.keras.models.Model(inputs=x_input, outputs=x, name="embedding")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=cnf.model.optimizers.adam.learning_rate),
        loss=tfa.losses.TripletSemiHardLoss())

    logs = os.path.join(p, "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir=logs, histogram_freq=1, profile_batch='500,520')
    ]

    # Train the network
    history = model.fit(
        dataset,
        callbacks=callbacks,
        epochs=cnf.model.epochs)

    model.save(
        modelpath,
        save_format='h5',
        overwrite=True
    )


if __name__ == '__main__':
    input_shape, cnf, p, dataset, modelpath = setup()
    train(input_shape=input_shape,
          cnf=cnf,
          p=p,
          dataset=dataset,
          modelpath=modelpath)
