##############################################################################
##
## data_manipulation.py
##
## @author: Matthew Cline
## @version: 20190318
##
## Description: Transform the raw data from kaggle into consummable vectors
## from each team for each year. The vectors for each year should include 
## the following information:
##
##  TeamID
##  Year
##  AvgScore_Home
##  AvgScore_Away
##  AvgOppScore_Home
##  AvgOppScore_Away
##  WinPerct_Home
##  WinPerct_Away
##  FGPerct_Home
##  FGPerct_Away
##  OppFGPerct_Home
##  OppFGPerct_Away
##  3ptPerct_Home
##  3ptPerct_Away
##  Opp3ptPerct_Home
##  Opp3ptPerct_Away
##  FTPerct_Home
##  FTPerct_Away
##  OppFTPerct_Home
##  OppFTPerct_Away
##  OR_Home
##  OR_Away
##  OppOR_Home
##  OppOR_Away
##  DR_Home
##  DR_Away
##  OppDR_Home
##  OppDR_Away
##  Assist_Home
##  Assist_Away
##  OppAssist_Home
##  OppAssist_Away
##  TO_Home
##  TO_Away
##  OppTO_Home
##  OppTO_Away
##  Steal_Home
##  Steal_Away
##  OppSteal_Home
##  OppSteal_Away
##  Blk_Home
##  Blk_Away
##  OppBlk_Home
##  OppBlk_Away
##
## The final model will concatenate the vectors for each team to determine the
## winner of the game. Values should be scaled and normalized to bypass the
## scaling issues between the different stats. The exception to this is
## the Coach and Conference values as they are mappings not scalars.
##
##############################################################################

import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm

##############################################################################
## 
## Import raw data files from the PROJECT_HOME/data directory
##  Teams.csv = key value map of team names
##  RegularSeasonDetailedResults.csv = box score stats from every game from 
##      2003 to present
##
##############################################################################

teams = pd.read_csv(os.path.normpath("data/stage_2/Teams.csv"))
box_scores = pd.read_csv(os.path.normpath("data/stage_2/RegularSeasonDetailedResults.csv"))



##############################################################################
##
## Build the dataframe to hold the yearly stats for each team
##
##############################################################################

first_team = np.min(teams['TeamID'])
last_team = np.max(teams['TeamID'])
num_teams = last_team - first_team + 1
first_season = np.min(box_scores['Season'])
last_season = np.max(box_scores['Season'])
num_seasons = last_season - first_season + 1

iterables = [np.linspace(start=first_team, stop=last_team, num=num_teams, dtype=int),
         np.linspace(start=first_season, stop=last_season, num=num_seasons, dtype=int)]

index = pd.MultiIndex.from_product(iterables, names=['team', 'year'])

yearly_stats = pd.DataFrame(data=np.zeros((num_teams*num_seasons,68)), index=index, columns=[
    'score_home',
    'score_away',
    'opp_score_home',
    'opp_score_away',
    'win_perct_home',
    'win_perct_away',
    'fg_perct_home',
    'fg_made_home',
    'fg_att_home',
    'fg_perct_away',
    'fg_made_away',
    'fg_att_away',
    'opp_fg_perct_home',
    'opp_fg_made_home',
    'opp_fg_att_home',
    'opp_fg_perct_away',
    'opp_fg_made_away',
    'opp_fg_att_away',
    '3pt_perct_home',
    '3pt_fg_made_home',
    '3pt_fg_att_home',
    '3pt_perct_away',
    '3pt_fg_made_away',
    '3pt_fg_att_away',
    'opp_3pt_perct_home',
    'opp_3pt_fg_made_home',
    'opp_3pt_fg_att_home',
    'opp_3pt_perct_away',
    'opp_3pt_fg_made_away',
    'opp_3pt_fg_att_away',
    'ft_perct_home',
    'ft_made_home',
    'ft_att_home',
    'ft_perct_away',
    'ft_made_away',
    'ft_att_away',
    'opp_ft_perct_home',
    'opp_ft_made_home',
    'opp_ft_att_home',
    'opp_ft_perct_away',
    'opp_ft_made_away',
    'opp_ft_att_away',
    'or_home',
    'or_away',
    'opp_or_home',
    'opp_or_away',
    'dr_home',
    'dr_away',
    'opp_dr_home',
    'opp_dr_away',
    'ast_home',
    'ast_away',
    'opp_ast_home',
    'opp_ast_away',
    'to_home',
    'to_away',
    'opp_to_home',
    'opp_to_away',
    'stl_home',
    'stl_away',
    'opp_stl_home',
    'opp_stl_away',
    'blk_home',
    'blk_away',
    'opp_blk_home',
    'opp_blk_away',
    'games_home',
    'games_away'])

##############################################################################
##
## Update the yearly stat lines for each team based on the regular
## season detailed box scores.
##
##############################################################################


for index, row in tqdm(box_scores.iterrows(), total=box_scores.shape[0]):

    season = row['Season']
    win_team = row['WTeamID']
    lose_team = row['LTeamID']

    if row['WLoc'] == 'H':

        ### Update the winning teams home stats
        yearly_stats.loc[(win_team,season), 'score_home'] += row['WScore']
        yearly_stats.loc[(win_team,season), 'opp_score_home'] += row['LScore']
        yearly_stats.loc[(win_team,season), 'win_perct_home'] += 1
        yearly_stats.loc[(win_team,season), 'fg_made_home'] += row['WFGM']
        yearly_stats.loc[(win_team,season), 'fg_att_home'] += row['WFGA']
        yearly_stats.loc[(win_team,season), 'opp_fg_made_home'] += row['LFGM']
        yearly_stats.loc[(win_team,season), 'opp_fg_att_home'] += row['LFGA']
        yearly_stats.loc[(win_team,season), '3pt_fg_made_home'] += row['WFGM3']
        yearly_stats.loc[(win_team,season), '3pt_fg_att_home'] += row['WFGA3']
        yearly_stats.loc[(win_team,season), 'opp_3pt_fg_made_home'] += row['LFGM3']
        yearly_stats.loc[(win_team,season), 'opp_3pt_fg_att_home'] += row['LFGA3']
        yearly_stats.loc[(win_team,season), 'ft_made_home'] += row['WFTM']
        yearly_stats.loc[(win_team,season), 'ft_att_home'] += row['WFTA']
        yearly_stats.loc[(win_team,season), 'opp_ft_made_home'] += row['LFTM']
        yearly_stats.loc[(win_team,season), 'opp_ft_att_home'] += row['LFTA']
        yearly_stats.loc[(win_team,season), 'or_home'] += row['WOR']
        yearly_stats.loc[(win_team,season), 'opp_or_home'] += row['LOR']
        yearly_stats.loc[(win_team,season), 'dr_home'] += row['WDR']
        yearly_stats.loc[(win_team,season), 'opp_dr_home'] += row['LDR']
        yearly_stats.loc[(win_team,season), 'ast_home'] += row['WAst']
        yearly_stats.loc[(win_team,season), 'opp_ast_home'] += row['LAst']
        yearly_stats.loc[(win_team,season), 'to_home'] += row['WTO']
        yearly_stats.loc[(win_team,season), 'opp_to_home'] += row['LTO']
        yearly_stats.loc[(win_team,season), 'stl_home'] += row['WStl']
        yearly_stats.loc[(win_team,season), 'opp_stl_home'] += row['LStl']
        yearly_stats.loc[(win_team,season), 'blk_home'] += row['WBlk']
        yearly_stats.loc[(win_team,season), 'opp_blk_home'] += row['LBlk']
        yearly_stats.loc[(win_team,season), 'games_home'] += 1

        ### Update the losing teams away stats ###
        yearly_stats.loc[(lose_team,season), 'score_away'] += row['LScore']
        yearly_stats.loc[(lose_team,season), 'opp_score_away'] += row['WScore']
        yearly_stats.loc[(lose_team,season), 'win_perct_away'] += 0
        yearly_stats.loc[(lose_team,season), 'fg_made_away'] += row['LFGM']
        yearly_stats.loc[(lose_team,season), 'fg_att_away'] += row['LFGA']
        yearly_stats.loc[(lose_team,season), 'opp_fg_made_away'] += row['WFGM']
        yearly_stats.loc[(lose_team,season), 'opp_fg_att_away'] += row['WFGA']
        yearly_stats.loc[(lose_team,season), '3pt_fg_made_away'] += row['LFGM3']
        yearly_stats.loc[(lose_team,season), '3pt_fg_att_away'] += row['LFGA3']
        yearly_stats.loc[(lose_team,season), 'opp_3pt_fg_made_away'] += row['WFGM3']
        yearly_stats.loc[(lose_team,season), 'opp_3pt_fg_att_away'] += row['WFGA3']
        yearly_stats.loc[(lose_team,season), 'ft_made_away'] += row['LFTM']
        yearly_stats.loc[(lose_team,season), 'ft_att_away'] += row['LFTA']
        yearly_stats.loc[(lose_team,season), 'opp_ft_made_away'] += row['WFTM']
        yearly_stats.loc[(lose_team,season), 'opp_ft_att_away'] += row['WFTA']
        yearly_stats.loc[(lose_team,season), 'or_away'] += row['LOR']
        yearly_stats.loc[(lose_team,season), 'opp_or_away'] += row['WOR']
        yearly_stats.loc[(lose_team,season), 'dr_away'] += row['LDR']
        yearly_stats.loc[(lose_team,season), 'opp_dr_away'] += row['WDR']
        yearly_stats.loc[(lose_team,season), 'ast_away'] += row['LAst']
        yearly_stats.loc[(lose_team,season), 'opp_ast_away'] += row['WAst']
        yearly_stats.loc[(lose_team,season), 'to_away'] += row['LTO']
        yearly_stats.loc[(lose_team,season), 'opp_to_away'] += row['WTO']
        yearly_stats.loc[(lose_team,season), 'stl_away'] += row['LStl']
        yearly_stats.loc[(lose_team,season), 'opp_stl_away'] += row['WStl']
        yearly_stats.loc[(lose_team,season), 'blk_away'] += row['LBlk']
        yearly_stats.loc[(lose_team,season), 'opp_blk_away'] += row['WBlk']
        yearly_stats.loc[(lose_team,season), 'games_away'] += 1

    elif row['WLoc'] == 'A':

        ### Update the winning teams away stats
        yearly_stats.loc[(win_team,season), 'score_away'] += row['WScore']
        yearly_stats.loc[(win_team,season), 'opp_score_away'] += row['LScore']
        yearly_stats.loc[(win_team,season), 'win_perct_away'] += 1
        yearly_stats.loc[(win_team,season), 'fg_made_away'] += row['WFGM']
        yearly_stats.loc[(win_team,season), 'fg_att_away'] += row['WFGA']
        yearly_stats.loc[(win_team,season), 'opp_fg_made_away'] += row['LFGM']
        yearly_stats.loc[(win_team,season), 'opp_fg_att_away'] += row['LFGA']
        yearly_stats.loc[(win_team,season), '3pt_fg_made_away'] += row['WFGM3']
        yearly_stats.loc[(win_team,season), '3pt_fg_att_away'] += row['WFGA3']
        yearly_stats.loc[(win_team,season), 'opp_3pt_fg_made_away'] += row['LFGM3']
        yearly_stats.loc[(win_team,season), 'opp_3pt_fg_att_away'] += row['LFGA3']
        yearly_stats.loc[(win_team,season), 'ft_made_away'] += row['WFTM']
        yearly_stats.loc[(win_team,season), 'ft_att_away'] += row['WFTA']
        yearly_stats.loc[(win_team,season), 'opp_ft_made_away'] += row['LFTM']
        yearly_stats.loc[(win_team,season), 'opp_ft_att_away'] += row['LFTA']
        yearly_stats.loc[(win_team,season), 'or_away'] += row['WOR']
        yearly_stats.loc[(win_team,season), 'opp_or_away'] += row['LOR']
        yearly_stats.loc[(win_team,season), 'dr_away'] += row['WDR']
        yearly_stats.loc[(win_team,season), 'opp_dr_away'] += row['LDR']
        yearly_stats.loc[(win_team,season), 'ast_away'] += row['WAst']
        yearly_stats.loc[(win_team,season), 'opp_ast_away'] += row['LAst']
        yearly_stats.loc[(win_team,season), 'to_away'] += row['WTO']
        yearly_stats.loc[(win_team,season), 'opp_to_away'] += row['LTO']
        yearly_stats.loc[(win_team,season), 'stl_away'] += row['WStl']
        yearly_stats.loc[(win_team,season), 'opp_stl_away'] += row['LStl']
        yearly_stats.loc[(win_team,season), 'blk_away'] += row['WBlk']
        yearly_stats.loc[(win_team,season), 'opp_blk_away'] += row['LBlk']
        yearly_stats.loc[(win_team,season), 'games_away'] += 1

        ### Update the losing teams home stats ###
        yearly_stats.loc[(lose_team,season), 'score_home'] += row['LScore']
        yearly_stats.loc[(lose_team,season), 'opp_score_home'] += row['WScore']
        yearly_stats.loc[(lose_team,season), 'win_perct_home'] += 0
        yearly_stats.loc[(lose_team,season), 'fg_made_home'] += row['LFGM']
        yearly_stats.loc[(lose_team,season), 'fg_att_home'] += row['LFGA']
        yearly_stats.loc[(lose_team,season), 'opp_fg_made_home'] += row['WFGM']
        yearly_stats.loc[(lose_team,season), 'opp_fg_att_home'] += row['WFGA']
        yearly_stats.loc[(lose_team,season), '3pt_fg_made_home'] += row['LFGM3']
        yearly_stats.loc[(lose_team,season), '3pt_fg_att_home'] += row['LFGA3']
        yearly_stats.loc[(lose_team,season), 'opp_3pt_fg_made_home'] += row['WFGM3']
        yearly_stats.loc[(lose_team,season), 'opp_3pt_fg_att_home'] += row['WFGA3']
        yearly_stats.loc[(lose_team,season), 'ft_made_home'] += row['LFTM']
        yearly_stats.loc[(lose_team,season), 'ft_att_home'] += row['LFTA']
        yearly_stats.loc[(lose_team,season), 'opp_ft_made_home'] += row['WFTM']
        yearly_stats.loc[(lose_team,season), 'opp_ft_att_home'] += row['WFTA']
        yearly_stats.loc[(lose_team,season), 'or_home'] += row['LOR']
        yearly_stats.loc[(lose_team,season), 'opp_or_home'] += row['WOR']
        yearly_stats.loc[(lose_team,season), 'dr_home'] += row['LDR']
        yearly_stats.loc[(lose_team,season), 'opp_dr_home'] += row['WDR']
        yearly_stats.loc[(lose_team,season), 'ast_home'] += row['LAst']
        yearly_stats.loc[(lose_team,season), 'opp_ast_home'] += row['WAst']
        yearly_stats.loc[(lose_team,season), 'to_home'] += row['LTO']
        yearly_stats.loc[(lose_team,season), 'opp_to_home'] += row['WTO']
        yearly_stats.loc[(lose_team,season), 'stl_home'] += row['LStl']
        yearly_stats.loc[(lose_team,season), 'opp_stl_home'] += row['WStl']
        yearly_stats.loc[(lose_team,season), 'blk_home'] += row['LBlk']
        yearly_stats.loc[(lose_team,season), 'opp_blk_home'] += row['WBlk']
        yearly_stats.loc[(lose_team,season), 'games_home'] += 1

    else:

        ### Update both teams away stats for a neutral site game
        yearly_stats.loc[(win_team,season), 'score_away'] += row['WScore']
        yearly_stats.loc[(win_team,season), 'opp_score_away'] += row['LScore']
        yearly_stats.loc[(win_team,season), 'win_perct_away'] += 1
        yearly_stats.loc[(win_team,season), 'fg_made_away'] += row['WFGM']
        yearly_stats.loc[(win_team,season), 'fg_att_away'] += row['WFGA']
        yearly_stats.loc[(win_team,season), 'opp_fg_made_away'] += row['LFGM']
        yearly_stats.loc[(win_team,season), 'opp_fg_att_away'] += row['LFGA']
        yearly_stats.loc[(win_team,season), '3pt_fg_made_away'] += row['WFGM3']
        yearly_stats.loc[(win_team,season), '3pt_fg_att_away'] += row['WFGA3']
        yearly_stats.loc[(win_team,season), 'opp_3pt_fg_made_away'] += row['LFGM3']
        yearly_stats.loc[(win_team,season), 'opp_3pt_fg_att_away'] += row['LFGA3']
        yearly_stats.loc[(win_team,season), 'ft_made_away'] += row['WFTM']
        yearly_stats.loc[(win_team,season), 'ft_att_away'] += row['WFTA']
        yearly_stats.loc[(win_team,season), 'opp_ft_made_away'] += row['LFTM']
        yearly_stats.loc[(win_team,season), 'opp_ft_att_away'] += row['LFTA']
        yearly_stats.loc[(win_team,season), 'or_away'] += row['WOR']
        yearly_stats.loc[(win_team,season), 'opp_or_away'] += row['LOR']
        yearly_stats.loc[(win_team,season), 'dr_away'] += row['WDR']
        yearly_stats.loc[(win_team,season), 'opp_dr_away'] += row['LDR']
        yearly_stats.loc[(win_team,season), 'ast_away'] += row['WAst']
        yearly_stats.loc[(win_team,season), 'opp_ast_away'] += row['LAst']
        yearly_stats.loc[(win_team,season), 'to_away'] += row['WTO']
        yearly_stats.loc[(win_team,season), 'opp_to_away'] += row['LTO']
        yearly_stats.loc[(win_team,season), 'stl_away'] += row['WStl']
        yearly_stats.loc[(win_team,season), 'opp_stl_away'] += row['LStl']
        yearly_stats.loc[(win_team,season), 'blk_away'] += row['WBlk']
        yearly_stats.loc[(win_team,season), 'opp_blk_away'] += row['LBlk']
        yearly_stats.loc[(win_team,season), 'games_away'] += 1

        yearly_stats.loc[(lose_team,season), 'score_away'] += row['LScore']
        yearly_stats.loc[(lose_team,season), 'opp_score_away'] += row['WScore']
        yearly_stats.loc[(lose_team,season), 'win_perct_away'] += 0
        yearly_stats.loc[(lose_team,season), 'fg_made_away'] += row['LFGM']
        yearly_stats.loc[(lose_team,season), 'fg_att_away'] += row['LFGA']
        yearly_stats.loc[(lose_team,season), 'opp_fg_made_away'] += row['WFGM']
        yearly_stats.loc[(lose_team,season), 'opp_fg_att_away'] += row['WFGA']
        yearly_stats.loc[(lose_team,season), '3pt_fg_made_away'] += row['LFGM3']
        yearly_stats.loc[(lose_team,season), '3pt_fg_att_away'] += row['LFGA3']
        yearly_stats.loc[(lose_team,season), 'opp_3pt_fg_made_away'] += row['WFGM3']
        yearly_stats.loc[(lose_team,season), 'opp_3pt_fg_att_away'] += row['WFGA3']
        yearly_stats.loc[(lose_team,season), 'ft_made_away'] += row['LFTM']
        yearly_stats.loc[(lose_team,season), 'ft_att_away'] += row['LFTA']
        yearly_stats.loc[(lose_team,season), 'opp_ft_made_away'] += row['WFTM']
        yearly_stats.loc[(lose_team,season), 'opp_ft_att_away'] += row['WFTA']
        yearly_stats.loc[(lose_team,season), 'or_away'] += row['LOR']
        yearly_stats.loc[(lose_team,season), 'opp_or_away'] += row['WOR']
        yearly_stats.loc[(lose_team,season), 'dr_away'] += row['LDR']
        yearly_stats.loc[(lose_team,season), 'opp_dr_away'] += row['WDR']
        yearly_stats.loc[(lose_team,season), 'ast_away'] += row['LAst']
        yearly_stats.loc[(lose_team,season), 'opp_ast_away'] += row['WAst']
        yearly_stats.loc[(lose_team,season), 'to_away'] += row['LTO']
        yearly_stats.loc[(lose_team,season), 'opp_to_away'] += row['WTO']
        yearly_stats.loc[(lose_team,season), 'stl_away'] += row['LStl']
        yearly_stats.loc[(lose_team,season), 'opp_stl_away'] += row['WStl']
        yearly_stats.loc[(lose_team,season), 'blk_away'] += row['LBlk']
        yearly_stats.loc[(lose_team,season), 'opp_blk_away'] += row['WBlk']
        yearly_stats.loc[(lose_team,season), 'games_away'] += 1




##############################################################################
##
## Calculate the percentages and per game metrics for each team/season
##
##############################################################################

for idx, row in tqdm(yearly_stats.iterrows(), total=yearly_stats.shape[0]):

    ### Update percentages for each team/season ###
    if row['fg_att_home']:
        yearly_stats.loc[idx, 'fg_perct_home'] = row['fg_made_home'] / row['fg_att_home']

    if row['fg_att_away']:
        yearly_stats.loc[idx, 'fg_perct_away'] = row['fg_made_away'] / row['fg_att_away']

    if row['opp_fg_att_home']:
        yearly_stats.loc[idx, 'opp_fg_perct_home'] = row['opp_fg_made_home'] / row['opp_fg_att_home']

    if row['opp_fg_att_away']:
        yearly_stats.loc[idx, 'opp_fg_perct_away'] = row['opp_fg_made_away'] / row['opp_fg_att_away']

    if row['3pt_fg_att_home']:
        yearly_stats.loc[idx, '3pt_perct_home'] = row['3pt_fg_made_home'] / row['3pt_fg_att_home']

    if row['3pt_fg_att_away']:
        yearly_stats.loc[idx, '3pt_perct_away'] = row['3pt_fg_made_away'] / row['3pt_fg_att_away']

    if row['opp_3pt_fg_att_home']:
        yearly_stats.loc[idx, 'opp_3pt_perct_home'] = row['opp_3pt_fg_made_home'] / row['opp_3pt_fg_att_home']

    if row['opp_3pt_fg_att_away']:
        yearly_stats.loc[idx, 'opp_3pt_perct_away'] = row['opp_3pt_fg_made_away'] / row['opp_3pt_fg_att_away']

    if row['ft_att_home']:
        yearly_stats.loc[idx, 'ft_perct_home'] = row['ft_made_home'] / row['ft_att_home']

    if row['ft_att_away']:
        yearly_stats.loc[idx, 'ft_perct_away'] = row['ft_made_away'] / row['ft_att_away']

    if row['opp_ft_att_home']:
        yearly_stats.loc[idx, 'opp_ft_perct_home'] = row['opp_ft_made_home'] / row['opp_ft_att_home']

    if row['opp_ft_att_away']:
        yearly_stats.loc[idx, 'opp_ft_perct_away'] = row['opp_ft_made_away'] / row['opp_ft_att_away']

    if row['games_home']:
        yearly_stats.loc[idx, 'score_home'] = row['score_home'] / row['games_home']
        yearly_stats.loc[idx, 'opp_score_home'] = row['opp_score_home'] / row['games_home']
        yearly_stats.loc[idx, 'win_perct_home'] = row['win_perct_home'] / row['games_home']
        yearly_stats.loc[idx, 'or_home'] = row['or_home'] / row['games_home']
        yearly_stats.loc[idx, 'opp_or_home'] = row['opp_or_home'] / row['games_home']
        yearly_stats.loc[idx, 'dr_home'] = row['dr_home'] / row['games_home']
        yearly_stats.loc[idx, 'opp_dr_home'] = row['opp_dr_home'] / row['games_home']
        yearly_stats.loc[idx, 'ast_home'] = row['ast_home'] / row['games_home']
        yearly_stats.loc[idx, 'opp_ast_home'] = row['opp_ast_home'] / row['games_home']
        yearly_stats.loc[idx, 'to_home'] = row['to_home'] / row['games_home']
        yearly_stats.loc[idx, 'opp_to_home'] = row['opp_to_home'] / row['games_home']
        yearly_stats.loc[idx, 'stl_home'] = row['stl_home'] / row['games_home']
        yearly_stats.loc[idx, 'opp_stl_home'] = row['opp_stl_home'] / row['games_home']
        yearly_stats.loc[idx, 'blk_home'] = row['blk_home'] / row['games_home']
        yearly_stats.loc[idx, 'opp_blk_home'] = row['opp_blk_home'] / row['games_home']

    if row['games_away']:
        yearly_stats.loc[idx, 'score_away'] = row['score_away'] / row['games_away']
        yearly_stats.loc[idx, 'opp_score_away'] = row['opp_score_away'] / row['games_away']
        yearly_stats.loc[idx, 'win_perct_away'] = row['win_perct_away'] / row['games_away']
        yearly_stats.loc[idx, 'or_away'] = row['or_away'] / row['games_away']
        yearly_stats.loc[idx, 'opp_or_away'] = row['opp_or_away'] / row['games_away']
        yearly_stats.loc[idx, 'dr_away'] = row['dr_away'] / row['games_away']
        yearly_stats.loc[idx, 'opp_dr_away'] = row['opp_dr_away'] / row['games_away']
        yearly_stats.loc[idx, 'ast_away'] = row['ast_away'] / row['games_away']
        yearly_stats.loc[idx, 'opp_ast_away'] = row['opp_ast_away'] / row['games_away']
        yearly_stats.loc[idx, 'to_away'] = row['to_away'] / row['games_away']
        yearly_stats.loc[idx, 'opp_to_away'] = row['opp_to_away'] / row['games_away']
        yearly_stats.loc[idx, 'stl_away'] = row['stl_away'] / row['games_away']
        yearly_stats.loc[idx, 'opp_stl_away'] = row['opp_stl_away'] / row['games_away']
        yearly_stats.loc[idx, 'blk_away'] = row['blk_away'] / row['games_away']
        yearly_stats.loc[idx, 'opp_blk_away'] = row['opp_blk_away'] / row['games_away']


##############################################################################
##
## Dump the final dataframe to pickle object file
##
##############################################################################

pickle.dump(yearly_stats, open(os.path.normpath("custom_data/yearly_stats.p"), "wb"))
