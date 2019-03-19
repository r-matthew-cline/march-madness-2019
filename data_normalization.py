##############################################################################
##
## data_normalization.py
##
## @author: Matthew Cline
## @version: 20190319
##
## Description: Normalize a pandas dataframe that is stored as a pickled 
## object. The object is passed in as an argument to the program.
##
##############################################################################

import pandas as pd
import pickle
import sys.argv as args
import os

if len(args) < 3:
    print("You must pass an input file as the first argument and an export file as the second argument...")
    exit(1)

import_path = os.path.normpath(args[1])
export_path = os.path.normpath(args[2])
df = pickle.load(open(import_path, "rb"))

normalized_df = (df - df.min())/(df.max() - df.min())

pickle.dump(normalized_df, open(export_path, "wb"))
