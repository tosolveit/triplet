import tensorflow as tf
import tensorflow_addons as tfa
import ruamel.yaml
from box import Box
from src.processinput import ProcessInput
from datetime import datetime
from src.gempoolinglayer import GeMPoolingLayer

# get configuration file
cnf = Box.from_yaml(filename="../params.yaml", Loader=ruamel.yaml.Loader)

datadir = 'data/train'

# create tensorflow dataset object
processinput = ProcessInput(data_dir=datadir,
                            num_classes_per_batch=cnf.model.num_classes_per_batch,
                            num_images_per_class=cnf.model.num_image_per_class,
                            channels=cnf.image.channels,
                            img_width=cnf.image.width,
                            img_height=cnf.image.height,
                            file_extension=cnf.image.extension)
dataset = processinput.train_input_fn()


reg = tf.keras.regularizers
input_shape = (cnf.image.height, cnf.image.width, cnf.image.channels)
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
    x = tf.keras.layers.Dense(cnf.model.embeddingsize, activation='softplus', kernel_regularizer=reg.l2(), dtype='float32')(x)

    model = tf.keras.models.Model(inputs=x_input, outputs=x, name="embedding")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=cnf.optimizers.adam.learning_rate),
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
        epochs=cnf.model.epochs)


model.save(
    cnf.model.name,
    save_format='h5',
    overwrite=True
)