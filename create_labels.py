##############################################################################
##
## create_labels.py
##
## @author: Matthew Cline
## @version: 20190319
##
## Description: Creates a training set by iterating through the regular
## season game data to build a new dataframe with the following attributes
##      home_team
##      away_team
##      season
##      home_team_outcome
##      away_team_outcome
## the first three values will be used to pull the appropriate record from
## the yearly stats dataframe, and the last two serve as the labels for the 
## model.
##
##############################################################################

import numpy as np
import pickle
import pandas as pd
import os
from tqdm import tqdm

### Read in data from csv ###
raw_data = pd.read_csv(os.path.normpath("data/stage_2/RegularSeasonDetailedResults.csv"))
team_data = pickle.load(open(os.path.normpath("custom_data/yearly_stats_normalized.p"), "rb"))
data = []
labels = []
val_data = []
val_labels = []

### Reformat the data into new dataframe ###
for idx, row in tqdm(raw_data.iterrows(), total=raw_data.shape[0]):
    if row['WLoc'] == 'H':
        season = row['Season']
        home_team = row['WTeamID']
        away_team = row['LTeamID']
        if season < 2018:
            labels.append(np.array([1.0, 0.0]))
            labels.append(np.array([0.0, 1.0]))
        else:
            val_labels.append(np.array([1.0, 0.0]))
            val_labels.append(np.array([0.0, 1.0]))
    else:
        season = row['Season']
        away_team = row['WTeamID']
        home_team = row['LTeamID']
        if season < 2018:
            labels.append(np.array([0.0, 1.0]))
            labels.append(np.array([1.0, 0.0]))
        else:
            val_labels.append(np.array([0.0, 1.0]))
            val_labels.append(np.array([1.0, 0.0]))

    if season < 2018:
        data.append(np.concatenate((team_data.loc[(home_team, season)].values,
            team_data.loc[(away_team, season)].values)))
        data.append(np.concatenate((team_data.loc[(away_team, season)].values,
            team_data.loc[(home_team, season)].values)))
    else:
        val_data.append(np.concatenate((team_data.loc[(home_team, season)].values,
            team_data.loc[(away_team, season)].values)))
        val_data.append(np.concatenate((team_data.loc[(away_team, season)].values,
            team_data.loc[(home_team, season)].values)))

data = np.array(data)
labels = np.array(labels)
val_data = np.array(val_data)
val_labels = np.array(val_labels)

print("data.shape = ", data.shape)
print("labels.shape = ", labels.shape)
print("val_data.shape = ", val_data.shape)
print("val_labels.shape = ", val_labels.shape)

### Dump data and labels to pickle objects ###
pickle.dump(data, open(os.path.normpath("train_data/data.p"), "wb"))
pickle.dump(labels, open(os.path.normpath("train_data/labels.p"), "wb"))
pickle.dump(val_data, open(os.path.normpath("val_data/data.p"), "wb"))
pickle.dump(val_labels, open(os.path.normpath("val_data/labels.p"), "wb"))
