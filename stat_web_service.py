import pandas as pd
import pickle
import numpy as np
import os

from flask import Flask
from flask import jsonify

df = pickle.load(open(os.path.normpath("custom_data/yearly_stats.p"), "rb"))


app = Flask(__name__)

@app.route('/get-stats/<int:team_id>', methods=['GET'])
def get_stats(team_id):
    out = {
        "Score": df.loc[(team_id, 2019), 'score_home'],
        "OppScore": df.loc[(team_id, 2019), 'opp_score_home'],
        "OffReb": df.loc[(team_id, 2019), 'or_home'],
        "DefReb": df.loc[(team_id, 2019), 'dr_home'],
        "Ast": df.loc[(team_id, 2019), 'ast_home'],
        "Stl": df.loc[(team_id, 2019), 'stl_home'],
        "Blk": df.loc[(team_id, 2019), 'blk_home']
    }
    return jsonify(out)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5100)

