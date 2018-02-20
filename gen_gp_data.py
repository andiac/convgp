import keras
from keras.datasets import mnist
from keras import backend as K

from keras.models import Model
from keras.models import load_model

import os
import numpy as np
import pickle
import scipy.io as sio
import myutil

if __name__ == "__main__":
    layer_names = ['dropout_2', 'flatten_1', 'dense_2']
    dataset_names = ['mnist', 'mnist_with_awgn', 'mnist_with_motion_blur', 'mnist_with_reduced_contrast_and_awgn']

    test_list = []
    train_list = []

    model = load_model('model12.h5')

    for dataset_name in dataset_names:
        X, Y, Xt, Yt = myutil.get_dataset(dataset_name)
        test = dict()
        test["dataset_name"] = dataset_name
        test["layer_name"] = 'raw'
        test["Xt"] = Xt
        test["Yt"] = Yt.astype("int64")
        test["feature"] = np.array([Xt[idx].flatten() for idx in range(len(Xt))])
        print(dataset_name, 'raw')
        print(Xt.shape)
        print(Yt.shape)
        print(test["feature"].shape)
        test_list.append(test)
        for layer_name in layer_names:
            sub_model = myutil.get_sub_model(model, layer_name)
            feature = sub_model.predict(Xt)
            print(dataset_name, layer_name)
            print(Xt.shape)
            print(Yt.shape)
            print(feature.shape)
            test = dict()
            test["dataset_name"] = dataset_name
            test["layer_name"] = layer_name
            test["Xt"] = Xt
            test["Yt"] = Yt.astype("int64")
            test["feature"] = feature
            test_list.append(test)

    for test in test_list:
        pickle.dump(test, open(os.path.join("./myres", test["dataset_name"] + "-" + test["layer_name"] + '.test'), "wb"))

    X, Y, Xt, Yt = myutil.get_dataset('mnist')
    train = dict()
    train["layer_name"] = 'raw'
    train["X"] = np.array([X[idx].flatten() for idx in range(len(X))])
    train["Y"] = Y.astype("int64")
    train["Xt"] = np.array([Xt[idx].flatten() for idx in range(len(Xt))])
    train["Yt"] = Yt.astype("int64")
    train_list.append(train)
    print(train['layer_name'])
    print(train["X"].shape)
    print(train["Y"].shape)
    for layer_name in layer_names:
        sub_model = myutil.get_sub_model(model, layer_name)
        feature = sub_model.predict(Xt)
        train = dict()
        train["layer_name"] = layer_name
        train["X"] = sub_model.predict(X)
        train["Y"] = Y.astype("int64")
        train["Xt"] = sub_model.predict(Xt)
        train["Yt"] = Yt.astype("int64")
        train_list.append(train)
        print(train['layer_name'])
        print(train["X"].shape)
        print(train["Y"].shape)

    for train in train_list:
        pickle.dump(train, open(os.path.join("./myres", train["layer_name"] + '.train'), "wb"))
