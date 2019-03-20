##############################################################################
##
## prediction.py
##
## @author: Matthew Cline
## @version: 20190320
##
## Description: Using the model trained to pick the winner of a single
## basketball game, this program will return the name of the winner
## between the two teams provided as arguments.
##
##############################################################################

import pickle
import os
from keras.models import Sequential, model_from_yaml
from keras.layers import Dense, LeakyReLU, Activation
import tensorflow as tf
import numpy as np
import pandas as pd
import sys

if len(sys.argv) != 3:
    print("You must enter two teams to predict the winner...")
    exit(1)

### Load the team mappings ###
teams = pd.read_csv(os.path.normpath("data/stage_2/TeamSpellings.csv"), encoding='latin1')
teams = teams.set_index('TeamNameSpelling')
stats = pickle.load(open(os.path.normpath("custom_data/yearly_stats_normalized.p"), "rb"))

### Get team id ###
team1 = 0
team2 = 0
try:
    team1 = teams.loc[sys.argv[1], 'TeamID']
except:
    print("Unable to find Team1. Check spelling...")
    exit(1)
try:
    team2 = teams.loc[sys.argv[2], 'TeamID']
except:
    print("Unable to find Team2. Check spelling...")

### Create input vector ###
input_vector = np.array([np.concatenate((stats.loc[(team1, 2019)].values, stats.loc[(team2, 2019)].values))])

### Load the model architecture and weights ###
model_file = open(os.path.normpath("models/20190319.yaml"), "r")
model_config = model_file.read()
model_file.close()
model = model_from_yaml(model_config)

model.load_weights(os.path.normpath("weights/20190319.h5"))
print("Model successfully loaded.\n\n")

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy', 'mean_squared_error'])

pred = model.predict(input_vector)
print("Predictions: \n\t%s: %f\n\t%s: %f" % (sys.argv[1], pred[0][0], sys.argv[2], pred[0][1]))
