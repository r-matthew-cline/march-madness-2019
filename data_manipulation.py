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
##  Conference
##  Coach
##  Seed
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

##############################################################################
## 
## Import raw data files from the PROJECT_HOME/data directory
##  Teams.csv = key value map of team names
##  RegularSeasonDetailedResults.csv = box score stats from every game from 
##      2003 to present
##  TeamCoaches = coach mappings to year and team
##  NCAATourneySeeds.csv = Team mapping to tournament seed and year
##  TeamConferences.csv = Team mapping to conference and year
##
##############################################################################

teams = pd.read_csv(os.path.normpath("data/Teams.csv"))
box_scores = pd.read_csv(os.path.normpath("data/RegularSeasonDetailedResults.csv"))
coaches = pd.read_csv(os.path.normpath("data/TeamCoaches.csv"))
rankings = pd.read_csv(os.path.normpath("data/NCAATourneySeeds.csv"))
conferences = pd.read_csv(os.path.normpath("data/TeamConferences.csv"))

### Confirm files read successfully ###
#print("Teams: \n", teams[0:10])
#print("\nBox Scores: \n", box_scores[0:10])
#print("\nCoaches: \n", coaches[0:10])
#print("\nRankings: \n", rankings[0:10])
#print("\nConferences: \n", conferences[0:10])

first_team = np.min(teams['TeamID'])
last_team = np.max(teams['TeamID'])
num_teams = last_team - first_team
first_season = np.min(box_scores['Season'])
last_season = np.max(box_scores['Season'])
num_seasons = last_season - first_season

iterables = [np.linspace(start=first_team, stop=last_team, num=num_teams, dtype=int),
         np.linspace(start=first_season, stop=last_season, num=num_seasons, dtype=int)]

index = pd.MultiIndex.from_product(iterables, names=['team', 'year'])

yearly_stats = pd.DataFrame(data=np.zeros((num_teams*num_seasons,44)), index=index, columns=[
    'score_home',
    'score_away',
    'opp_score_home',
    'opp_score_away',
    'win_perct_home',
    'win_perct_away',
    'fg_perct_home',
    'fg_perct_away',
    'opp_fg_perct_home',
    'opp_fg_perct_away',
    '3pt_perct_home',
    '3pt_perct_away',
    'opp_3pt_perct_home',
    'opp_3pt_perct_away',
    'ft_perct_home',
    'ft_perct_away',
    'opp_ft_perct_home',
    'opp_ft_perct_away',
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

print(yearly_stats)
