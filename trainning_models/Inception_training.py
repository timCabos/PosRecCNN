##########################################################
# CNN training code from Luca Doria, using data_crafter
# files Keras Tuner and different Charge normailations ?
##########################################################

from tensorflow import keras
import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
#from kerastuner.tuners import RandomSearch

from keras.models import Sequential, model_from_json, Model
from keras.layers import Flatten, Dense, Dropout, Activation, Input, Conv2D, MaxPooling2D, BatchNormalization
from keras.constraints import max_norm, MinMaxNorm
from keras.layers.merge import concatenate

#from sklearn.utils import shuffle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import os, sys

def inception_module(layer_in, f1, f2, f3):
 # 1x1 conv
 conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
 # 3x3 conv
 conv3 = Conv2D(f2, (3,3), padding='same', activation='relu')(layer_in)
 # 5x5 conv
 conv5 = Conv2D(f3, (5,5), padding='same', activation='relu')(layer_in)
 # 3x3 max pooling
 pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
 # concatenate filters, assumes filters/channels last
 layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1) #-1 -> concatenate in the last dimension
 return layer_out


def projected_inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
        # 1x1 conv
        conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)

        # 3x3 conv
        conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(layer_in)
        conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)

        # 5x5 conv
        conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(layer_in)
        conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)

        # 3x3 max pooling
        pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
        pool = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)

        # concatenate filters, assumes filters/channels last
        layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
        return layer_out


def residual_module(layer_in, n_filters):
         merge_input = layer_in
         # check if the number of filters needs to be increase, assumes channels last format
         if layer_in.shape[-1] != n_filters:
          merge_input = Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
         # conv1
         conv1 = Conv2D(n_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
         # conv2
         conv2 = Conv2D(n_filters, (3,3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
         # add filters, assumes filters/channels last
         layer_out = add([conv2, merge_input])
         # activation function
         layer_out = Activation('relu')(layer_out)
         return layer_out


## Load training data ###################################

#fq = open('../data/All_events_Q_training.dat','r')
#fy = open('../data/All_data_coordinates_training.dat','r')
#fq = open('/Users/tim/Desktop/DEAP_3600/Codes/Inception/my_data/10kevt/events_pmt_Q_TRAIN_from_test2bis_10k_ntp.dat','r')#
fq = np.loadtxt('PMT_Q_TRAIN_file_from_data_crafter.py', delimiter=' ')
fy = np.loadtxt('events_mc_position_TRAIN_file_from_data_crafter.py', delimiter = ' ')
#fy = open('events_mc_position_TRAIN_from_test2bis_10k_ntp.dat','r')
## Augmented coordinates are basically the same as non-augmented. I just had some errors popping up so I
## decided to go with separate variables and loading

## Settings #############################################
#events = len(fy.readlines())
events = len(fy[:,0])
px = 16
py = 16
clr = 1

X_train = np.ndarray(shape=(events,px,py,clr))
Y_train = np.ndarray(shape=(events,3))

## Load input matrices ######################################

print('Reading matrices...')
'''
l=0
ln=0
j=0
for line in fq.readlines():
     line = line.strip()
     col = line.split() #list

     i=0

     for cl in col:
         if (float(cl)>0.001):
             X_train[l][i][j] = float(cl)
         else:
             X_train[l][i][j] = 0
         i = i+1

     ln = ln +1

     j = j+1

     if (ln%px==0):
         l = l+1
         j=0
'''
X_train = np.reshape(fq, (events,16,16))
print("Charge matrices loaded...", X_train)

## Load outputs #############################################

Y_train = (fy/840.0 + 1)*0.5
print("Position test matrices loaded..", Y_train)

'''
i=0
l=0
for line in fy.readlines():
     line = line.strip()
     col = line.split()
     i=0
     for cl in col:
         Y_train[l][i] = (float(cl)/840.0 + 1)*0.5
         i=i+1

     l=l+1

print("Coordinates loaded...", Y_train)
## Load augmented matrices ##################################
'''
'''
l=15000 ## starts where the loading of ordinary data stoppped
ln=0
j=0
for line in fqa.readlines():
     line = line.strip()
     col = line.split() #list

     i=0

     for cl in col:
        X_train[l][i][j] = float(cl) ## no need to condition. Done while augmenting.
        i = i+1

     ln = ln +1

     j = j+1

     if (ln%px==0):
         l = l+1
         j=0

print("Augmented charge matrices loaded...", X_train)

## Load outputs ############################################

i=0
l=15000 ## starts where the loading of ordinary data stoppped
for line in fya.readlines():
     line = line.strip()
     col = line.split()
     i=0
     for cl in col:
         Y_train[l][i] = (float(cl)/840.0 + 1)*0.5
         i=i+1

     l=l+1

print("Augmented coordinates loaded...", Y_train)
'''
## Shuffle data ############################################

#X_train, Y_train = shuffle(X_train, Y_train)

# This is done so that the validation set doesn't bias towards
# augmented data while training.

## Model ###################################################

model = Sequential()

#model.add(Conv2D(128, kernel_size=(2,2), activation='relu', strides=(2,2), input_shape=(256,256))) #3,3 1,1 relu
#model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2))) #2,2

#inception_module(layer_in, f1, f2, f3):
#input_shape = Input(shape=(16, 16, 3))
#model.add(inception_module(input_shape, 64, 64, 64))
#model.add(Flatten())
#model.add(Dense(64, activation='relu', kernel_constraint=max_norm(3)))
#model.add(Dropout(0.3))
#model.add(Dense(3, activation='sigmoid'))

input_shape = Input(shape=(16, 16, 1))
out = inception_module(input_shape, 32, 32, 32)
out = BatchNormalization()(out)
#out = projected_inception_module(input_shape, 32, 32, 16, 32, 16, 16)
out = Dropout(0.3)(out)
#out = inception_module(out, 32, 32, 32)
#out = Dropout(0.3)(out)
out = Flatten()(out)
out = Dense(200, activation="sigmoid")(out)
out = Dropout(0.3)(out)
out = Dense(200, activation="sigmoid")(out)
out = Dropout(0.3)(out)
out = Dense(200, activation="sigmoid")(out)
out = Dropout(0.3)(out)
out = Dense(3, activation="sigmoid")(out)
model = Model(inputs=input_shape, outputs=out)



#opt = keras.optimizers.Adam(learning_rate=0.000032)
#model.compile(loss='mse',optimizer=opt, metrics=['mse'])

model.compile(loss='mse',optimizer='SGD', metrics=['mse'])

## Training ################################################

history = model.fit(X_train, Y_train, batch_size = 256,
                    validation_split = 0.9, epochs = 3)

predictions = model.predict(X_train)

############################################################

# Save model to JSON
model_json = model.to_json()
with open("model_FFNN_my.json", "w") as json_file:
     json_file.write(model_json)
#Save weights to HDF5
model.save_weights("model_FFNN_my.h5")
print("Saved model to disk")

# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)

plt.savefig('loss_ffnn.pdf')
