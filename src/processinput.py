import tensorflow as tf
import pathlib
import numpy as np

# from tensorflow.contrib.data.python.ops.interleave_ops import DirectedInterleaveDataset
class ProcessInput:
    def __init__(self,
                 data_dir,
                 num_classes_per_batch=8,
                 num_images_per_class=4,
                 channels=3,
                 img_width=120,
                 img_height=120,
                 file_extension='png'):
        self.data_dir = data_dir
        self.num_classes_per_batch = num_classes_per_batch
        self.num_images_per_class = num_images_per_class
        self.channels = channels
        self.img_width = img_width
        self.img_height = img_height
        self.file_extension = file_extension

    def train_input_fn(self):
        data_root = pathlib.Path(self.data_dir)
        # all_image_paths = list(data_root.glob('**/*.jpeg'))
        all_image_paths = list(data_root.glob('**/*.{}'.format(self.file_extension)))
        all_directories = {'/'.join(str(i).split("/")[:-1]) for i in all_image_paths}
        print("-----")
        print("num of labels: ")
        print(len(all_directories))
        print("-----")
        labels_index = list(i.split("/")[-1] for i in all_directories)

        # Create the list of datasets creating filenames
        # datasets = [tf.data.Dataset.list_files("{}/*.jpeg".format(image_dir), shuffle=False) for image_dir in
        #            all_directories]
        datasets = [tf.data.Dataset.list_files("{}/*.{}".format(image_dir, self.file_extension), shuffle=False) for
                    image_dir in
                    all_directories]

        num_labels = len(all_directories)
        num_classes_per_batch = self.num_classes_per_batch
        num_images_per_class = self.num_images_per_class

        def get_label_index(s):
            return labels_index.index(s.numpy().decode("utf-8").split("/")[-2])

        # def preprocess_image(image):
        #   image = tf.cast(image, tf.float32)
        #   image = tf.math.divide(image, 255.0)
        #   return image
        #
        # def load_and_preprocess_image(path):
        #     image = tf.io.read_file(path)
        #     return tf.py_function(preprocess_image, [image], tf.float32), tf.py_function(get_label_index, [path], tf.int64)

        def load_and_preprocess_image(path):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=self.channels)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize(image, [self.img_height, self.img_width])
            # return tf.data.Dataset.from_tensors((image, tf.py_function(get_label_index, [path], tf.int64)))
            return image, tf.py_function(get_label_index, [path], tf.int64)

        def generator():
            while True:
                # Sample the labels that will compose the batch
                labels = np.random.choice(range(num_labels),
                                          num_classes_per_batch,
                                          replace=False)

                for label in labels:
                    for _ in range(num_images_per_class):
                        yield label

        choice_dataset = tf.data.Dataset.from_generator(generator, tf.int64)
        dataset = tf.data.experimental.choose_from_datasets(datasets, choice_dataset)

        dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # dataset = dataset.interleave(load_and_preprocess_image, cycle_length=2,num_parallel_calls=tf.data.experimental.AUTOTUNE)

        batch_size = num_classes_per_batch * num_images_per_class
        print("----------------------")
        print('batch_size: %s' % batch_size)
        print("----------------------")
        dataset = dataset.batch(batch_size)
        # dataset = dataset.repeat(self.params.num_epochs)
        dataset = dataset.prefetch(1)

        # print(dataset)
        return dataset