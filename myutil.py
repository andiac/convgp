from keras.datasets import mnist
from keras.models import Model
from keras.models import load_model
from keras import backend as K

import os
import numpy as np
import scipy.io as sio

def get_dataset(dataset_name):
    img_rows, img_cols = 28, 28

    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()   

        if K.image_data_format() == 'channels_first':
            x_train= x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_train /= 255
        x_test = x_test.astype('float32')
        x_test /= 255

        return (x_train, y_train, x_test, y_test)

    else:
        filename = dataset_name.replace('_', '-') + ".mat"

        data = sio.loadmat(os.path.join("./datasets", filename))
        x_train = data["train_x"]
        x_test = data["test_x"]
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols) 
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_train /= 255
        y_train = np.argmax(data["train_y"], axis=1)
        x_test = x_test.astype('float32')
        x_test /= 255
        y_test = np.argmax(data["test_y"], axis=1)

        return (x_train, y_train, x_test, y_test)

def get_sub_model(model, layer_name):
    featureExtractorModel = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    return featureExtractorModel
