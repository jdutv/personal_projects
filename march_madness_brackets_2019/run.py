import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import sys

def transform_data(full_data):
    data = full_data[(full_data['forecast_date'] == '2019-03-19') & (full_data['gender'] == 'mens') & (full_data['team_alive'] == 1)][['team_region', 'team_seed', 'team_name', 'team_rating', 'rd2_win']].rename(columns={'rd2_win': 'rd1_win'})
    data['team_seed'] = data['team_seed'].str.replace('[a-z]', '').astype(int)
    data['rd1_matchup'] = (abs(8.5 - data['team_seed']) * 2 + 1) / 2
    data['rd2_matchup'] = (abs(4.5 - data['rd1_matchup']) * 2 + 1) / 2
    data['rd3_matchup'] = (abs(2.5 - data['rd2_matchup']) * 2 + 1) / 2
    data['rd4_matchup'] = (abs(1.5 - data['rd3_matchup']) * 2 + 1) / 2
    data['rd5_matchup'] = 1
    data.loc[data['team_region'].isin(['South', 'Midwest']), 'rd5_matchup'] = 2
    data['rd6_matchup'] = 1

    return data

def train_model(data, seed):
    matchup_key = ['team_region', 'rd1_matchup']
    model_data = form_matchup(data, matchup_key)
    model_data['upper_rating_pct'] = model_data['team_rating_upper'] / (model_data['team_rating_upper'] + model_data['team_rating_lower'])
    full_x = model_data.loc[:, ['upper_rating_pct']].values
    full_y = model_data.loc[:, 'rd1_win_upper'].values
    model = RandomForestRegressor(n_estimators=100, random_state = seed)
    model.fit(full_x, full_y)
    return model

def form_matchup(data, matchup_key):
    model_data = data.merge(data, on = matchup_key, suffixes = ('_upper', '_lower'))
    model_data = model_data[model_data['team_name_upper'] != model_data['team_name_lower']]
    model_data['better_team'] = model_data.apply(lambda x: x['team_name_upper'] if x['team_rating_upper'] >= x['team_rating_lower'] else x['team_name_lower'],axis = 1)
    model_data = model_data[model_data['team_name_upper'] == model_data['better_team']].drop('better_team', axis = 1).reset_index(drop = True)
    return model_data

def play(matchup):
    return np.random.choice([matchup['team_name_upper'], matchup['team_name_lower']], size = 1, p = [matchup['upper_win_prob'], 1-matchup['upper_win_prob']])[0]

def print_results(matchup, matchup_key):
    return '''({0:.0f}) {1} [{2:.2f}] vs ({3:.0f}) {4} [{5:.2f}]\n{1} Win Prob: {6:.2%}\tProj. Winner: {7}'''.format(
             matchup['team_seed_upper'], matchup['team_name_upper'], matchup['team_rating_upper'],
             matchup['team_seed_lower'], matchup['team_name_lower'], matchup['team_rating_lower'],
             matchup['upper_win_prob'], matchup['winner'])

if __name__ == '__main__':

    seed = int(sys.argv[1])
    np.random.seed(seed)

    full_data = pd.read_csv('fivethirtyeight_ncaa_forecasts.csv')
    data = transform_data(full_data)

    model = train_model(data, seed)

    for rd in range(1, 7):
        matchup_key = ['rd{}_matchup'.format(rd)]
        if rd < 5:
            matchup_key = ['team_region'] + matchup_key

        if rd == 1:
            round_data = data
        else:
            round_data = data[data['team_name'].isin(surviving_teams)]

        round_matchup = form_matchup(round_data, matchup_key)
        round_matchup['upper_rating_pct'] = round_matchup['team_rating_upper'] / (round_matchup['team_rating_upper'] + round_matchup['team_rating_lower'])

        pred_x = round_matchup.loc[:, ['upper_rating_pct']].values
        round_matchup['upper_win_prob'] = model.predict(pred_x)

        np.random.seed(seed)
        round_matchup['winner'] = round_matchup.apply(lambda a: play(a), axis = 1)
        if 'team_region' in list(round_matchup.columns):
            round_matchup = round_matchup.sort_values(['team_region', 'team_seed_upper']).reset_index(drop = True)
        else:
            round_matchup = round_matchup.sort_values(['team_seed_upper', 'team_name_upper']).reset_index(drop = True)

        surviving_teams = list(round_matchup['winner'])

        print('==== ROUND {} ===='.format(rd))
        for i in range(round_matchup.shape[0]):
            print(print_results(round_matchup.loc[i, :], matchup_key))
    else:
        total_points = np.random.randint(120, 150)
        winner_points = int(np.ceil(total_points * round_matchup['upper_win_prob']))
        loser_point = total_points - winner_points
        print('Tiebreaker: {0} - {1}'.format(winner_points, loser_point))
        print('==== CHAMPION: {} ===='.format(surviving_teams[0].upper()))
