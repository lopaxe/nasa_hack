import tensorflow as tf
import numpy as np
import PIL.Image
import cv2 as cv


class FullTransformation:

    def __init__(self, original_path, style_path, output_path, params, model):
        self.original_path = original_path
        self.style_path = style_path
        self.output_path = output_path
        self.params = params
        self.model = model

    @staticmethod
    def tensor_to_image(tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)

    @staticmethod
    def load_img(path_to_img):
        max_dim = 512
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    @property
    def content_image(self):
        return self.load_img(self.original_path)

    @property
    def full_transformation(self):

        style_image = self.load_img(self.style_path)
        return self.tensor_to_image(self.model(tf.constant(self.content_image), tf.constant(style_image))[0][0])

    @property
    def original_transformation(self):
        return self.tensor_to_image(self.content_image)


class WeightedTransformation:

    def __init__(self, original_path, full_transformation_path, output_path, params):
        self.original_path = original_path
        self.full_transformation_path = full_transformation_path
        self.output_path = output_path
        self.params = params
        if params["T_wt"]/2 != 0:
            self.beta = 0.5 + params["T_wt"]/2
        else:
            self.beta = 0

    @property
    def original_image(self):
        return cv.imread(self.original_path)

    @property
    def styled_image(self):
        return cv.imread(self.full_transformation_path)

    @property
    def weighted_image(self):
        alpha = 1-self.beta
        return cv.addWeighted(self.original_image, alpha, self.styled_image, self.beta, 0.0)
