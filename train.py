import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_addons as tfa


from processinput import ProcessInput
from helpers import extractzip
from datetime import datetime


class params:
    num_classes_per_batch = 8
    num_images_per_class = 4
    num_epochs = 3
    img_height = 120
    img_width = 120
    channels = 3

extractzip('train.zip','train')
datadir = 'train'
train_func = ProcessInput(data_dir=datadir, params=params, file_extension='jpeg')
dataset = train_func.train_input_fn()
input_shape = (params.img_width, params.img_height)

# train
class GeMPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, p=1., eps=1e-6):
        super().__init__()
        self.p = p
        self.eps = eps

    def call(self, inputs: tf.Tensor, **kwargs):
        inputs = tf.clip_by_value(
            inputs,
            clip_value_min=self.eps,
            clip_value_max=tf.reduce_max(inputs)
        )
        inputs = tf.pow(inputs, self.p)
        inputs = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)
        inputs = tf.pow(inputs, 1. / self.p)

        return inputs

    def get_config(self):
        return {
            'p': self.p,
            'eps': self.eps
        }


reg = tf.keras.regularizers
input_shape = (params.img_height, params.img_width, 3)
embeddingsize = 256
with tf.device('/gpu:0'):
    imagenet = tf.keras.applications.Xception(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False)

    imagenet.trainable = False

    # embedding model
    x_input = tf.keras.Input(shape=input_shape)
    x = imagenet(x_input)
    x = GeMPoolingLayer()(x)
    x = tf.keras.layers.Dense(embeddingsize, activation='softplus', kernel_regularizer=reg.l2(), dtype='float32')(x)

    model = tf.keras.models.Model(inputs=x_input, outputs=x, name="embedding")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0001),
        loss=tfa.losses.TripletSemiHardLoss())

    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir=logs, histogram_freq=1, profile_batch='500,520')
    ]

    # Train the network
    history = model.fit(
        dataset,
        callbacks=callbacks,
        epochs=50)


model.save(
    'model-rijk.h5',
    save_format='h5',
    overwrite=True
)

