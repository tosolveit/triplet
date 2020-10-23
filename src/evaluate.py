import tensorflow as tf
import numpy as np
from annoy import AnnoyIndex
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
import os
from tqdm import tqdm
import random
from box import Box
import ruamel.yaml
from pathlib import Path
from gempoolinglayer import GeMPoolingLayer


class Evaluate:
    def __init__(self,
                 params_file_name='params.yaml',
                 images_folder_name='data/pix512',
                 images_test_folder_name='data/test',
                 project_root=(Path(__file__) / ".." / ".."),
                 custom_objects={'GeMPoolingLayer': GeMPoolingLayer},
                 sample_size=4
                 ):
        """

        :param params_file_name: str
            stores the parameters that is used for accessing values, in the constructor
            we obtain cnf object that is easy to access attributes like objects
            example: self.cnf.model.name will return name of the model
        :param images_folder_name:
        :param images_test_folder_name:
        :param project_root:
        :param custom_objects: dict
            when loading tensorflow models with the customer objects we have to pass this to
            tf.keras.models.load_model
        :param sample_size: int
            data/test folder has sub-folders one for each class, this parameter will define
            the number of images to sample for that class due to efficiency
        """
        # define the config and data paths
        self.params_file_name = params_file_name
        self.images_folder_name = images_folder_name
        self.images_test_folder_name = images_test_folder_name
        self.project_root = project_root
        self.custom_objects = custom_objects
        self.sample_size = sample_size

        # initialize params.yaml file
        self.params_path = project_root.joinpath(self.params_file_name).resolve()
        self.images_folder = project_root.joinpath(self.images_folder_name).iterdir()
        self.images_test_folder = project_root.joinpath(self.images_test_folder_name).iterdir()
        self.cnf = Box.from_yaml(filename=self.params_path, Loader=ruamel.yaml.Loader)

        # initialize index object
        self.t = AnnoyIndex(self.cnf.model.embeddingsize, 'euclidean')
        self.__number_of_classes = len(dir(self.images_test_folder))

    def load_model(self):
        """

        :return: returns a tensorflow model
        """
        model = tf.keras.models.load_model(
            self.cnf.model.name,
            custom_objects=self.custom_objects
        )
        return model

        # add predicted embedding values to index for the original images
    def read_image_and_convert_to_array(self, img_path):
        index = str(img_path).rsplit('/', 1)[-1].split('.')[0]
        img = Image.open(img_path)
        img = img.resize((self.cnf.image.width, self.cnf.image.height), Image.ANTIALIAS)
        img = img_to_array(img) / 255.
        return index, img

    def create_index(self):
        model = self.load_model()
        for img_path in self.images_folder:
            index, img = self.read_image_and_convert_to_array(img_path=img_path)
            predicted_embedding = model.predict(tf.expand_dims(img, axis=0)).flatten()
            self.t.add_item(int(index), predicted_embedding)
        return self.t

    def save_index(self, index_name='annoy_index_file'):
        index_object = self.create_index()
        index_object.build(self.cnf.annoy.ntrees)
        index_object.save(index_name)
        print(f"index saved to: {self.cnf.annoy.index_name}")

    def load_index(self):
        # load annoy index
        u = AnnoyIndex(self.cnf.model.embeddingsize, self.cnf.annoy.similiarity)
        u.load(self.cnf.annoy.index_name)  # super fast
        print(f"index loaded as: {self.cnf.annoy.index_name}")
        return u

    @staticmethod
    def _get_list_of_the_files_in_a_directory(directory):
        return list(directory.glob('*'))

    def sample_from_directory(self, directory):
        images_in_class = Evaluate._get_list_of_the_files_in_a_directory(directory=directory)
        samples = random.sample(images_in_class, self.sample_size)
        return samples

    def calculate_metric(self):
        model, index_base = self.load_model(), self.load_index()
        total = self.__number_of_classes * self.sample_size
        counter = 0
        with tqdm(total=total) as pbar:
            for test_class in dir(self.images_test_folder):
                # skip if the items inside the class is less than sample size
                if len(dir(test_class)) < self.sample_size:
                    continue
                for sample in self.sample_from_directory(test_class):
                    index, img = self.read_image_and_convert_to_array(img_path=sample)
                    predicted_embedding = model.predict(tf.expand_dims(img, axis=0)).flatten()
                    result = index_base.get_nns_by_vector(predicted_embedding, 1)
                    if result[0] == int(test_class):
                        counter += 1
                    pbar.update(1)
        result = counter / float(total)
        return result

    def write_metric(self, metrics_file):
        with open(metrics_file, 'w') as outfile:
            metric_score = self.calculate_metric()
            outfile.write("precision @1: " + str(metric_score) + "\n")


if __name__ == '__main__':
    evaluate = Evaluate()
    evaluate.save_index(index_name=evaluate.cnf.annoy.index.name)
    evaluate.write_metric('metrics_file.txt')

