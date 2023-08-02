import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import zipfile
import PIL
from PIL import Image
import os
from keras.models import load_model
from keras.layers import BatchNormalization

def preprocess(image):
    image = tf.image.resize(image, [64,64]) / 255.0
    return image

def load_ml():
    model = load_model('MyCnn.h5')
    return model

def predict():
    model = model = load_model('MyCnn.h5')
    testImage = keras.preprocessing.image.load_img('gae.png', target_size=(64,64))
    imageArr = np.array(testImage)
    imageArr = imageArr/255
    imageArr = imageArr.reshape(-1,64,64,3)
    pred = model.predict(imageArr)
    print(np.argmax(pred))
    plt.imshow(imageArr[0])
    plt.show()

load_ml()
predict()







