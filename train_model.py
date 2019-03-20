##############################################################################
##
## train_model.py
##
## @author: Matthew Cline
## @version: 20190319
##
## Description: Fully connected neural network model trainer to pick the 
## winner in a basketball game. Uses the average stats from the entire 
## season as the input. Provides a one hot encoding of each teams
## probability of winning as the output.
##
##############################################################################

import numpy as np
import os
import pickle
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

### Import the game logs ###
train_data = pickle.load(open(os.path.normpath("train_data/data.p"), "rb"))
train_labels = pickle.load(open(os.path.normpath("train_data/labels.p"), "rb"))
val_data = pickle.load(open(os.path.normpath("val_data/data.p"), "rb"))
val_labels = pickle.load(open(os.path.normpath("val_data/labels.p"), "rb"))

### Model Architecture ###
model = Sequential()
model.add(Dense(256, input_dim=136))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(64))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(32))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(8))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(2))
model.add(Activation(tf.nn.softmax))

callbacks = [EarlyStopping(monitor='val_loss', patience=3),
             ModelCheckpoint(filepath=os.path.normpath("weights/best_model.h5"),
                             monitor='val_loss',
                             save_best_only=True)]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mean_squared_error'])
model.fit(train_features=train_data, 
          train_target=train_labels,
          epochs=150,
          batch_size=10,
          callbacks=callbacks,
          verbose=0,
          validation_data=(val_data,val_labels))

model_yaml =  model.to_yaml()
with open("models/20190320.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
print("Saved model config to models/20190320.yaml")
model.save_weights("weights/20190320.h5")
print("Saved model weights to weights/20190320.h5")

