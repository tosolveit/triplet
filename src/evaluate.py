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
from box import Box
import ruamel.yaml
from gempoolinglayer import GeMPoolingLayer


# get configuration file
cnf = Box.from_yaml(filename="config.yml", Loader=ruamel.yaml.Loader)

# load model
model = tf.keras.models.load_model(
     cnf.model.name,
     custom_objects={'GeMPoolingLayer': GeMPoolingLayer}
)


# write raw inputs to annoy index
extractzip('pix512.zip')
images_folder = 'pix512'

# initialize index object
t = AnnoyIndex(cnf.model.embeddingsize, 'euclidean')

# add predicted embedding values to index for the original images
for i in os.listdir(images_folder):
    index, extension  = i.split('.')
    if not extension == cnf.image.extension:
        continue
    img = Image.open(os.path.join(images_folder,i)) # image extension *.png,*.jpg
    img = img.resize((cnf.image.width, cnf.image.height), Image.ANTIALIAS)
    img = img_to_array(img) / 255.
    try:
        predicted_embedding = model.predict(tf.expand_dims(img,axis=0)).flatten()
    except ValueError:
        print(i)
        continue
    t.add_item(int(index), predicted_embedding)


# save annoy index
t.build(cnf.annoy.ntrees) # number of trees that will be build
t.save('rijk-embedding.ann')

# load annoy index
u = AnnoyIndex(cnf.model.embeddingsize, 'euclidean')
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
            result = u.get_nns_by_vector(predicted_embedding,1)  # folder is the
            if result[0] == int(folder):
                successfully_predict_counter+=1
            pbar.update(1)

print(successfully_predict_counter / 4323.)