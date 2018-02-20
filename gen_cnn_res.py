import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from keras.models import load_model
from keras.models import Model

import os
import numpy as np
import pickle
import myutil

m = load_model('model12.h5')

dataset_names = ['mnist', 'mnist_with_awgn', 'mnist_with_motion_blur', 'mnist_with_reduced_contrast_and_awgn']

for dataset_name in dataset_names:
    X, Y, Xt, Yt = myutil.get_dataset(dataset_name)
    
    prob = m.predict_proba(Xt)
    var = np.zeros(prob.shape)
    pred = np.argmax(prob, axis = 1)
    accuracy = np.sum(pred == Yt) * 1.0 / Yt.shape[0]
    
    res = dict()
    res["dataset_name"] = dataset_name
    res["Xt"] = Xt
    res["Yt"] = Yt.astype("int64")
    res["feature"] = np.array([Xt[idx].flatten() for idx in range(len(Xt))])
    res["prob"] = prob
    res["var"] = var
    res["accuracy"] = accuracy
    
    print(dataset_name, ":", accuracy)
    
    pickle.dump(res, open(os.path.join("./myres", dataset_name + ".test.res"), "wb"))
    
