##############################################################################
##
## kaggle_submission.py
##
## @author: Matthew Cline
## @version: 20190320
##
## Description: Model prediciton on all of the games for the kaggle
## competition.
##
##############################################################################

import pickle
import os
from keras.models import Sequential, model_from_yaml
from keras.layers import Dense, LeakyReLU, Activation
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm


### Load the team mappings ###
teams = pd.read_csv(os.path.normpath("data/stage_2/TeamSpellings.csv"), encoding='latin1')
teams = teams.set_index('TeamNameSpelling')
stats = pickle.load(open(os.path.normpath("custom_data/yearly_stats_normalized.p"), "rb"))
game_list = pd.read_csv(os.path.normpath("data/stage_2/SampleSubmissionStage2.csv"), encoding='latin1')


### Load the model architecture and weights ###
model_file = open(os.path.normpath("models/20190319.yaml"), "r")
model_config = model_file.read()
model_file.close()
model = model_from_yaml(model_config)

model.load_weights(os.path.normpath("weights/best_model.h5"))
print("Model successfully loaded.\n\n")

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy', 'mean_squared_error'])

### Loop through all games ###
for idx, row in tqdm(game_list.iterrows(), total=game_list.shape[0]):
    matchup = row['ID'].split("_")
    team1 = int(matchup[1])
    team2 = int(matchup[2])

    ### Create input vector ###
    input_vector = np.array([np.concatenate((stats.loc[(team1, 2019)].values, stats.loc[(team2, 2019)].values))])

    pred = model.predict(input_vector)
    game_list.loc[idx, 'Pred'] = pred[0][0]

game_list.to_csv(os.path.normpath("custom_data/kaggle_output.csv"), index=None, header=True)
