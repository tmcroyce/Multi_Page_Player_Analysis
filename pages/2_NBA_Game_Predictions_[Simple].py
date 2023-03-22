
# cd C:\Users\Travis\OneDrive\Data Science\Personal_Projects\Sports\NBA_Prediction_V3_1\notebooks_v3_1\4_Running_All

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import sqlite3
import seaborn as sns
from matplotlib.pyplot import figure
from bs4 import BeautifulSoup
import time
import requests     
import shutil       
import datetime
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')
import requests
import json
from random import randint
import  random
import os
import plotly.graph_objs as go
#os.chdir('C:\\Users\\Travis\\OneDrive\\Data Science\\Personal_Projects\\Sports\\NBA_Prediction_V3_1')
from cmath import nan
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import make_column_selector as selector, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
import pickle
from sklearn.metrics import fbeta_score
from sklearn.linear_model import LinearRegression
from sklearn import tree, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, f1_score, make_scorer, recall_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(layout='wide')

date_time = datetime.datetime.now()
# subtract 7 hours to get to PST
date_time = date_time - datetime.timedelta(hours=7)
today = date_time.strftime('%Y-%m-%d')

# Load
today_df = pd.read_csv('data/team/aggregates/daily_updates/today_df_'+str(today)+'.csv', low_memory=False)
today_df['home?'] = np.where(today_df['home_or_away'] == "home", 1, 0)
games_odds = pd.read_csv('data/team/aggregates/daily_updates/draftkings'+str(today)+'.csv', low_memory=False)
expected_minutes = pd.read_csv('data/team/aggregates/daily_updates/player_minutes_projection_{}.csv'.format(today))
expected_starters = pd.read_csv('data/team/aggregates/daily_updates/starting_lineups_'+str(today)+'.csv', low_memory=False)
player_metrics = pd.read_csv('data/player/aggregates_of_aggregates/trad_season_averages_imputed.csv')
player_advanced_metrics = pd.read_csv('data/team/aggregates/daily_updates/advanced_player_stats'+str(today)+'.csv', low_memory=False)
player_names = pd.read_csv('data/player/aggregates_of_aggregates/all_player_names.csv')

# D vs Position from today
d_vs_position = pd.read_csv('data/team/aggregates/daily_updates/defense_vs_position'+str(today)+'.csv', low_memory=False)
d_vs_position['PTS_rank'] = d_vs_position.groupby('POS')['PTS'].rank(ascending = False)
d_vs_position['PTS_rank'] = d_vs_position['PTS_rank'].astype(int)
keep_cols = ['TEAM', 'POS', 'PTS', 'REB', 'FG%', 'PTS_rank']
d_vs_position = d_vs_position[keep_cols]

# Player GbG
player_gbg = pd.read_csv('data/player/aggregates/Trad&Adv_box_scores_GameView.csv', low_memory=False)

# Load Team Metrics
team_boxscores = pd.read_csv('data/team/aggregates/All_Boxes.csv', low_memory=False)
# filter just this season
boxes_22 = team_boxscores[team_boxscores['trad_season'] == 2022]

# Current Usage Metrics
my_usage_metrics = pd.read_csv('data/player/aggregates/current_usg_metrics_'+ today +'.csv')

#player names
player_names = player_names[['first_initial_last_name', 'player_name']]

# fix up games_and_times
games_and_times = pd.read_csv('data/team/aggregates/NBA_Schedule.csv', low_memory=False)
games_and_times = games_and_times[games_and_times['Date'] == today]
games_and_times = games_and_times[['Start (ET)', 'Visitor/Neutral', 'Home/Neutral']] 

# minus 3 hours to get to PST
games_and_times['Start (ET)'] = pd.to_datetime(games_and_times['Start (ET)']) - datetime.timedelta(hours=3)
games_and_times['Start (ET)'] = games_and_times['Start (ET)'].dt.strftime('%I:%M %p')
games_and_times.rename(columns={'Start (ET)':'Time', 'Visitor/Neutral':'Away', 'Home/Neutral':'Home'}, inplace=True)


# load the initial predictions from the day
predictions = pd.read_csv('data/team/aggregates/daily_updates/predictions_today_'+str(today)+'.csv', low_memory=False)

# Predictions 2: Just load the final predictions after initial run
predictions_2 = pd.read_csv('data/team/aggregates/daily_updates/final_games_df_'+str(today)+'.csv')

team_to_abv = {'ATL Hawks': 'ATL',
                'BKN Nets': 'BKN',
                'BOS Celtics': 'BOS',
                'CHA Hornets': 'CHA',
                'CHI Bulls': 'CHI',
                'CLE Cavaliers': 'CLE',
                'DAL Mavericks': 'DAL',
                'DEN Nuggets': 'DEN',
                'DET Pistons': 'DET',
                'GS Warriors': 'GSW',
                'HOU Rockets': 'HOU',
                'IND Pacers': 'IND',
                'LA Clippers': 'LAC',
                'LA Lakers': 'LAL',
                'MEM Grizzlies': 'MEM',
                'MIA Heat': 'MIA',
                'MIL Bucks': 'MIL',
                'MIN Timberwolves': 'MIN',
                'NO Pelicans': 'NOP',
                'NY Knicks': 'NYK',
                'OKC Thunder': 'OKC',
                'ORL Magic': 'ORL',
                'PHI 76ers': 'PHI',
                'PHO Suns': 'PHX',
                'POR Trail Blazers': 'POR',
                'SAC Kings': 'SAC',
                'SA Spurs': 'SAS',
                'TOR Raptors': 'TOR',
                'UTA Jazz': 'UTA',
                'WAS Wizards': 'WAS'}

def fix_prediction(df):
        total_probs = df['probs'].sum()
        if total_probs > 1:
            # if the total probability is greater than 1, then we need to adjust the probabilities
            # get the difference between the total and 1
            df['diff'] = 1 / total_probs
            # multiply the probabilities by the difference
            df['new_probs'] = round(df['probs'] * df['diff'],2)
            # get the new total
            df['EV_Fixed'] = round((df['new_probs'] * df['CLOSING\nODDS']) - ((1-df['new_probs']) * 100),2)
            df['team_abv'] = df['TODAY'].map(team_to_abv)
            keep_columns = ['TODAY', 'team_abv', 'SPREAD', 'TOTAL', 'MONEYLINE', 'probs', 'prediction', 'EV', 'new_probs', 'EV_Fixed']
            df = df[keep_columns]
            return df

        elif total_probs < 1:
            # if the total probability is less than 1, then we need to adjust the probabilities
            # get the difference between the total and 1
            df['diff'] = 1 / total_probs
            # multiply the probabilities by the difference
            df['new_probs'] = round(df['probs'] * df['diff'],2)
            # get the new total
            df['EV_Fixed'] = round((df['new_probs'] * df['CLOSING\nODDS']) - ((1-df['new_probs']) * 100),2)
            df['team_abv'] = df['TODAY'].map(team_to_abv)
            keep_columns = ['TODAY', 'team_abv', 'SPREAD', 'TOTAL', 'MONEYLINE', 'probs', 'prediction', 'EV', 'new_probs', 'EV_Fixed']
            df = df[keep_columns]
            return df

        else:
            df['new_probs'] = df['probs'].round(2)
            df['EV_Fixed'] = round((df['new_probs'] * df['CLOSING\nODDS']) - ((1-df['new_probs']) * 100),2)
            df['team_abv'] = df['TODAY'].map(team_to_abv)
            keep_columns = ['TODAY', 'team_abv', 'SPREAD', 'TOTAL', 'MONEYLINE', 'probs', 'prediction', 'EV', 'new_probs', 'EV_Fixed']
            df = df[keep_columns]
            return df

def get_game(game_num):
        game_df = predictions.iloc[game_num*2:game_num*2+2]
        return game_df

# make sure probs and Closing odds are floats
predictions['CLOSING\nODDS'] = predictions['CLOSING\nODDS'].astype(str).str.strip()
predictions['CLOSING\nODDS'] = predictions['CLOSING\nODDS'].str.replace('+', '')
predictions['CLOSING\nODDS'] = predictions['CLOSING\nODDS'].str.replace('−', '-')

predictions['probs'] = predictions['probs'].astype(float)
predictions['CLOSING\nODDS'] = predictions['CLOSING\nODDS'].astype(float)

predictions['EV'] = (predictions['probs'] * predictions['CLOSING\nODDS']) - ((1-predictions['probs']) * 100)

g1 = get_game(0)
g1 = fix_prediction(g1)
g2 = get_game(1)
g2 = fix_prediction(g2)
g3 = get_game(2)
g3 = fix_prediction(g3)
g4 = get_game(3)
g4 = fix_prediction(g4)
g5 = get_game(4)
g5 = fix_prediction(g5)
g6 = get_game(5)
g6 = fix_prediction(g6)
g7 = get_game(6)
g7 = fix_prediction(g7)
g8 = get_game(7)
g8 = fix_prediction(g8)
g9 = get_game(8)
g9 = fix_prediction(g9)
g10 = get_game(9)
g10 = fix_prediction(g10)
g11 = get_game(10)
g11 = fix_prediction(g11)
g12 = get_game(11)
g12 = fix_prediction(g12)
games = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12]

games_df = pd.concat(games)
final_cols = ['TODAY', 'team_abv', 'SPREAD', 'TOTAL', 'MONEYLINE', 'new_probs']
games_df = games_df[final_cols]
# rename
games_df = games_df.rename(columns={'TODAY':'Team', 'SPREAD':'Spread', 'TOTAL':'Total', 'MONEYLINE':'Moneyline', 'new_probs':'Win Probability'})


# names dictionary
names_dict = {'ATL Hawks':'ATL',
                'BKN Nets':'BKN',
                'BOS Celtics':'BOS',
                'CHA Hornets':'CHA',
                'CHI Bulls':'CHI',
                'CLE Cavaliers':'CLE',
                'DAL Mavericks':'DAL',
                'DEN Nuggets':'DEN',
                'DET Pistons':'DET',
                'GS Warriors':'GSW',
                'HOU Rockets':'HOU',
                'IND Pacers':'IND',
                'LA Clippers':'LAC',
                'LA Lakers':'LAL',
                'MEM Grizzlies':'MEM',
                'MIA Heat':'MIA',
                'MIL Bucks':'MIL',
                'MIN Timberwolves':'MIN',
                'NO Pelicans':'NOP',
                'NY Knicks':'NYK',
                'OKC Thunder':'OKC',
                'ORL Magic':'ORL',
                'PHI 76ers':'PHI',
                'PHO Suns':'PHX',
                'POR Trail Blazers':'POR',
                'SAC Kings':'SAC',
                'SA Spurs':'SAS',
                'TOR Raptors':'TOR',
                'UTA Jazz':'UTA',
                'WAS Wizards':'WAS'}

games_df['path'] = 'data/team/logos/' + games_df['team_abv'] + '.png'



# load model
#model = pickle.load(open('pickle_models/pipe_1.pkl', 'rb'))


############ START OF APP ############

st.title('NBA Predictions with Machine Learning')
st.markdown('**Version 3.1**')
st.write('This app predicts the winner of NBA games using data from the current season and previous seasons.')
st.write('These predictions are based on probable lineups and can change if a player is ruled out or in.')


st.subheader(f'**Predictions at a glance**')
# find total number of games today
tot_games = len(games_odds)/2
tot_games = int(tot_games)

st.write("")

# if Spread is 'PK', change to 0
games_df['Spread'] = games_df['Spread'].replace('pk', 0)
# change spread to float
games_df['Spread'] = games_df['Spread'].astype(float)

games_df_show = games_df.drop(columns=['path'])

# add select button if they want to see a table with all games and teams
st.table(games_df_show.style.format({'Spread': '{:.1f}', 'Total': '{:.1f}', 'Win Probability': '{:.0%}', 'Moneyline': '{:+}'}))

st.write('Win Probability is the machine learning model\'s prediction of the probability of the team winning the game. EV is the expected value of a bet at the current odds with the model-predicted win probability.')

# Game 1
st.header(f'**Game by Game Predictions**')

# 2 columns
col1, col2 = st.columns(2)

# add path to image to games_df


# game 1 is first 2 rows of games_df
game_1 = games_df.iloc[0:2]
game_2 = games_df.iloc[2:4]
game_3 = games_df.iloc[4:6]
game_4 = games_df.iloc[6:8]
game_5 = games_df.iloc[8:10]
game_6 = games_df.iloc[10:12]
game_7 = games_df.iloc[12:14]
game_8 = games_df.iloc[14:16]
game_9 = games_df.iloc[16:18]
game_10 = games_df.iloc[18:20]
game_11 = games_df.iloc[20:22]
game_12 = games_df.iloc[22:24]

games_list = [game_1, game_2, game_3, game_4, game_5, game_6, game_7, game_8, game_9, game_10, game_11, game_12]


#all_games['path'] = 'data/team/logos/' + all_games['team_abv'] + '.png'

for game in games_list:
    if len(game) > 0:
        # put first team on left, second on right
        col1, col2 = st.columns(2)
        team1 = game['Team'].iloc[0]
        # write team1 vs team2
        col1.subheader(f'{team1} @ {game["Team"].iloc[1]}')
        col2.subheader(' ')

        col1.write(f'**{game["Team"].iloc[0]}**')
        col2.write(' ')
        col2.write(' ')
        col2.write(' ')
        
        # add team logo
        team1_logo_path = 'data/team/logos/' + game['team_abv'].iloc[0] + '.png'
        col1.image(team1_logo_path, width=100)

        team2_logo_path = 'data/team/logos/' + game['team_abv'].iloc[1] + '.png'
        col2.write(f'**{game["Team"].iloc[1]}**')
        col2.image(team2_logo_path, width=100)

        # Add Metrics
        col1.write(f'**Spread:** {game["Spread"].iloc[0]}')
        col2.write(f'**Spread:** {game["Spread"].iloc[1]}')
        col1.write(f'**Total:** {game["Total"].iloc[0]}')
        col2.write(f'**Total:** {game["Total"].iloc[1]}')
        col1.write(f'**Moneyline:** {game["Moneyline"].iloc[0]}')
        col2.write(f'**Moneyline:** {game["Moneyline"].iloc[1]}')
        col1.write(f'**Win Probability:** {game["Win Probability"].iloc[0]:.0%}')
        col2.write(f'**Win Probability:** {game["Win Probability"].iloc[1]:.0%}')

        # add space
        col1.write('')
        col2.write('')
        # add divider
        st.write('---')



st.write("")
st.write('There are {} games left to be played today.'.format(tot_games))
# get current time
now = datetime.datetime.now()
now = now.strftime("%H:%M:%S")
# change to normal time from military time
now = datetime.datetime.strptime(now, '%H:%M:%S').strftime('%I:%M %p')
# subtract 7 hours to get pst
now = datetime.datetime.strptime(now, '%I:%M %p') - datetime.timedelta(hours=7)
# st.sidebar.write('Last Refreshed: {}'.format(now))

st.sidebar.subheader('All Games Today')
st.sidebar.table(games_and_times.reset_index(drop=True))


games_odds['team_abv'] = games_odds['TODAY'].str.split(' ').str[0]
games_odds['team_name'] = games_odds['TODAY'].str.split(' ').str[1]
games_odds['team_fixed_abv'] = games_odds['TODAY'].map(names_dict)



# st.sidebar.subheader("Trav's Links")
# st.sidebar.markdown("[Lineups](https://www.rotowire.com/basketball/nba-lineups.php/)")
# st.sidebar.markdown("[On/Off](https://www.rotowire.com/basketball/court-on-off.php?team=BKN)")
# st.sidebar.markdown("[Teams](https://sports.yahoo.com/nba/teams/)")
# st.sidebar.markdown("[Odds](https://sportsbook.draftkings.com/leagues/basketball/nba)")

