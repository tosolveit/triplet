import tensorflow as tf
import numpy as np
from tensorflow import keras
from annoy import AnnoyIndex
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from helpers import extractzip
import os
import tqdm
import random

# load model
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

model = tf.keras.models.load_model(
     'model-rijk.h5',
     custom_objects={'GeMPoolingLayer': GeMPoolingLayer}
)


# write raw inputs to annoy index
new_width, new_height = 120, 120
extractzip('pix512.zip')
images_folder = 'pix512'

f = 256
t = AnnoyIndex(f, 'euclidean')

for i in os.listdir(images_folder):
    index, extension  = i.split('.')
    if not extension == 'jpeg':
        continue
    img = Image.open(os.path.join(images_folder,i)) # image extension *.png,*.jpg
    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    img = img_to_array(img) / 255.
    try:
        predicted_embedding = model.predict(tf.expand_dims(img,axis=0)).flatten()
    except ValueError:
        print(i)
        continue
    t.add_item(int(index), predicted_embedding)


# save annoy index
t.build(10) # 10 trees
t.save('rijk-embedding.ann')

# load annoy index
f = 256
u = AnnoyIndex(f, 'euclidean')
u.load('rijk-embedding.ann') # super fast


datadir = 'train'
error_counter = 0
successfully_predict_counter = 0
samp=4
with tqdm(total=4323*samp) as pbar:
    for folder in os.listdir(datadir):
        sample_dest = os.listdir(os.path.join(datadir,folder))
        if len(sample_dest) < samp:
            continue
        for img in random.sample(sample_dest,samp):
            image_path = os.path.join(datadir,folder,img)
            img = Image.open(image_path) # image extension *.png,*.jpg
            img = img_to_array(img) / 255.
            try:
                predicted_embedding = model.predict(tf.expand_dims(img,axis=0)).flatten()
            except:
                error_counter+=1
                continue
            result = u.get_nns_by_vector(predicted_embedding,1) # folder is the
            if result[0] == int(folder):
                successfully_predict_counter+=1
            pbar.update(1)

print(successfully_predict_counter / 4323.)