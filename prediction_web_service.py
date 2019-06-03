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

from flask import Flask
from flask import jsonify

### Load the team mappings ###
stats = pickle.load(open(os.path.normpath("custom_data/yearly_stats_normalized.p"), "rb"))


### Load the model architecture and weights ###
model_file = open(os.path.normpath("models/20190319.yaml"), "r")
model_config = model_file.read()
model_file.close()
model = model_from_yaml(model_config)

model.load_weights(os.path.normpath("weights/best_model.h5"))
print("Model successfully loaded.\n\n")

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy', 'mean_squared_error'])

model._make_predict_function()

app = Flask(__name__)


@app.route('/<int:team1>/<int:team2>')
def prediction(team1, team2, method='GET'):

    ### Create input vector ###
    input_vector = np.array([np.concatenate((stats.loc[(team1, 2019)].values, stats.loc[(team2, 2019)].values))])
    print(input_vector.shape)
    ### Make Prediction ###
    pred = model.predict(input_vector)
    out = {}
    out[str(team1)] = str(pred[0][0])
    out[str(team2)] = str(pred[0][1])
    print(out)
    return jsonify(out)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5200)
