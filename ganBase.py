from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np



# mnist image shape is 28x28 pixels
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)

# generator neurel network
def build_generator():

    # generator input is randome noise
    # vector of 1x100
    noise_shape = (100,)    

    #sequential inherits class model
    model = Sequential()

    # building generator layers
    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    model.summary()

    noise = Input(shape=noise_shape)
    # image generated 
    img = model(noise)    

    # return model
    return Model(noise, img)