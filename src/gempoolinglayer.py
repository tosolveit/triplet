import tensorflow as tf


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
