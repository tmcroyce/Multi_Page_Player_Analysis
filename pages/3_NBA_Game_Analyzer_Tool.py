
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


# testing... make predictions predictions2
predictions = predictions_2

# make sure probs and Closing odds are floats
predictions['CLOSING\nODDS'] = predictions['CLOSING\nODDS'].astype(str).str.strip()
predictions['CLOSING\nODDS'] = predictions['CLOSING\nODDS'].str.replace('+', '')
predictions['CLOSING\nODDS'] = predictions['CLOSING\nODDS'].str.replace('−', '-')

predictions['probs'] = predictions['probs'].astype(float)
predictions['CLOSING\nODDS'] = predictions['CLOSING\nODDS'].astype(float)

#predictions['EV'] = (predictions['probs'] * predictions['CLOSING\nODDS']) - ((1-predictions['probs']) * 100)

g1 = get_game(0)
# g1 = fix_prediction(g1)
g2 = get_game(1)
# g2 = fix_prediction(g2)
g3 = get_game(2)
# g3 = fix_prediction(g3)
g4 = get_game(3)
# g4 = fix_prediction(g4)
g5 = get_game(4)
# g5 = fix_prediction(g5)
g6 = get_game(5)
# g6 = fix_prediction(g6)
g7 = get_game(6)
# g7 = fix_prediction(g7)
g8 = get_game(7)
# g8 = fix_prediction(g8)
g9 = get_game(8)
# g9 = fix_prediction(g9)
g10 = get_game(9)
# g10 = fix_prediction(g10)
g11 = get_game(10)
# g11 = fix_prediction(g11)
g12 = get_game(11)
# g12 = fix_prediction(g12)
games = [g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12]

games_df = pd.concat(games)
final_cols = ['TODAY', 'trad_team', 'SPREAD', 'TOTAL', 'MONEYLINE', 'final_prob']
games_df = games_df[final_cols]
# rename
games_df = games_df.rename(columns={'TODAY':'Team', 'SPREAD':'Spread', 'TOTAL':'Total', 'MONEYLINE':'Moneyline', 'final_prob':'Win Probability'})


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

games_df['path'] = 'data/team/logos/' + games_df['trad_team'] + '.png'


# # add button
# button = st.sidebar.button('Refresh Expected Starters')

# # load model
# model = pickle.load(open('pickle_models/pipe_1.pkl', 'rb'))


############ START OF APP ############

st.title('NBA Predictions with Machine Learning')
st.markdown('**Version 3.1**')
st.write('This app predicts the winner of NBA games using data from the current season and previous seasons.')
st.write('These predictions are based on probable lineups and can change if a player is ruled out or in.')

# find total number of games today
tot_games = len(games_odds)/2
tot_games = int(tot_games)

st.write("")

# if Spread is 'PK', change to 0
games_df['Spread'] = games_df['Spread'].replace('pk', 0)
# change spread to float
games_df['Spread'] = games_df['Spread'].astype(float)

# add select button if they want to see a table with all games and teams
# if st.checkbox('Show All Games and Teams (one table)'):
#     st.table(games_df.style.format({'Spread': '{:.2f}', 'Total': '{:.2f}', 'Win Probability': '{:.0%}', 'EV': '{:.2f}', 'Moneyline': '{:+}'}))

# # Game 1
# st.subheader(f'**Game by Game Predictions**')

# # 2 columns
# col1, col2 = st.columns(2)

# # add path to image to games_df


# # game 1 is first 2 rows of games_df
# game_1 = games_df.iloc[0:2]
# game_2 = games_df.iloc[2:4]
# game_3 = games_df.iloc[4:6]
# game_4 = games_df.iloc[6:8]
# game_5 = games_df.iloc[8:10]
# game_6 = games_df.iloc[10:12]
# game_7 = games_df.iloc[12:14]
# game_8 = games_df.iloc[14:16]
# game_9 = games_df.iloc[16:18]
# game_10 = games_df.iloc[18:20]
# game_11 = games_df.iloc[20:22]
# game_12 = games_df.iloc[22:24]

# games_list = [game_1, game_2, game_3, game_4, game_5, game_6, game_7, game_8, game_9, game_10, game_11, game_12]


# #all_games['path'] = 'data/team/logos/' + all_games['team_abv'] + '.png'

# for game in games_list:
#     if len(game) > 0:
#         # put first team on left, second on right
#         col1, col2 = st.columns(2)
#         team1 = game['Team'].iloc[0]
#         # write team1 vs team2
#         col1.subheader(f'{team1} @ {game["Team"].iloc[1]}')
#         col2.subheader(' ')

#         col1.write(f'**{game["Team"].iloc[0]}**')
#         col2.write(' ')
#         col2.write(' ')
#         col2.write(' ')
        
#         # add team logo
#         team1_logo_path = 'data/team/logos/' + game['team_abv'].iloc[0] + '.png'
#         col1.image(team1_logo_path, width=100)

#         team2_logo_path = 'data/team/logos/' + game['team_abv'].iloc[1] + '.png'
#         col2.write(f'**{game["Team"].iloc[1]}**')
#         col2.image(team2_logo_path, width=100)

#         # Add Metrics
#         col1.write(f'**Spread:** {game["Spread"].iloc[0]}')
#         col2.write(f'**Spread:** {game["Spread"].iloc[1]}')
#         col1.write(f'**Total:** {game["Total"].iloc[0]}')
#         col2.write(f'**Total:** {game["Total"].iloc[1]}')
#         col1.write(f'**Moneyline:** {game["Moneyline"].iloc[0]}')
#         col2.write(f'**Moneyline:** {game["Moneyline"].iloc[1]}')
#         col1.write(f'**Win Probability:** {game["Win Probability"].iloc[0]:.0%}')
#         col2.write(f'**Win Probability:** {game["Win Probability"].iloc[1]:.0%}')
#         col1.write(f'**Expected Value:** {game["EV"].iloc[0]:.2f}')
#         col2.write(f'**Expected Value:** {game["EV"].iloc[1]:.2f}')

#         # add space
#         col1.write('')
#         col2.write('')
#         # add divider
#         st.write('---')



# st.write("")
# st.write('There are {} games left to be played today.'.format(tot_games))
# # get current time
# now = datetime.datetime.now()
# now = now.strftime("%H:%M:%S")
# # change to normal time from military time
# now = datetime.datetime.strptime(now, '%H:%M:%S').strftime('%I:%M %p')
# st.sidebar.write('Last Refreshed: {}'.format(now))

# st.sidebar.subheader('All Games Today')
# st.sidebar.table(games_and_times)

# show_plots = st.checkbox('Show Plots')

# if show_plots:
#     st.subheader('Expected Value and Win Probability')
#     st.write('The following plots show the expected value for a bet on each team and the win probability for each team.')
#     # plot Win Probability and EV with plotly
#     #fig = px.scatter(games_df, x='EV', y='Win Probability', color='Team', text='Team', color_discrete_sequence=px.colors.qualitative.Dark24)
#     #st.plotly_chart(fig)

#     games_df['path'] = 'data/team/logos/' + games_df['team_abv'] + '.png'

#     df = games_df
#     x = 'EV'
#     y = 'Win Probability'
#     fig, ax = plt.subplots(figsize=(8, 8), dpi=400)
#     ax.scatter(df[x], df[y])

#     # Change plot spines
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.spines['left'].set_color('#ccc8c8')
#     ax.spines['bottom'].set_color('#ccc8c8')

#     # Change ticks
#     plt.tick_params(axis='x', labelsize=12, color='#ccc8c8')
#     plt.tick_params(axis='y', labelsize=12, color='#ccc8c8')

#     # Plot images
#     def getImage(path):
#         return OffsetImage(plt.imread(path), zoom=.02, alpha = .8)

#     for index, row in df.iterrows():
#         ab = AnnotationBbox(getImage(row['path']), (row[x], row[y]), frameon=False)
#         ax.add_artist(ab)

#     # Add average lines
#     plt.hlines(df[y].mean(), df[x].min(), df[x].max(), color='#c2c1c0')
#     plt.vlines(0, df[y].min(), df[y].max(), color='#c2c1c0')


#     #Label
#     plt.title('EV vs Predicted Win Probability', size = 12)
#     plt.ylabel('Predicted Win Probability', size = 6)
#     plt.xlabel('Expected Value of $100', size = 6)

#     # fix height and width
#     st.pyplot(fig, use_container_width=False)

# else:
#     st.write('')
#     pass

# # add button to see predicted minutes
# button3 = st.sidebar.button('Refresh Odds')

# if button3:
#     # run the .py
#     os.system('python notebooks_v3_1/4_Running_All/draftkings_scrape.py')
#     # reload the data
#     games_odds = pd.read_csv('data/team/aggregates/daily_updates/draftkings'+str(today)+'.csv', low_memory=False)
#     st.write('Odds Updated')

games_odds['team_abv'] = games_odds['TODAY'].str.split(' ').str[0]
games_odds['team_name'] = games_odds['TODAY'].str.split(' ').str[1]
games_odds['team_fixed_abv'] = games_odds['TODAY'].map(names_dict)

a = 0
b = 2
n = 1
keyyy = 1
keyyyy = 100
skey = 20
wkey = 50

st.subheader('**Game Metrics**')



col1, col2 = st.columns(2)

# st.write('Select this box to see all games today.')
# see_all_games = col1.checkbox('See All Games?')

while n <= tot_games:
    # Display each game today
    st.subheader('Game #' + str(n)+ ': ' + games_odds['TODAY'].iloc[a] + ' @ ' + games_odds['TODAY'].iloc[a+1])

    # Locate Game
    game = games_odds.iloc[a:b]

    # Locate Teams
    team1 = game['team_fixed_abv'].iloc[0]
    team2 = game['team_fixed_abv'].iloc[1]
    display_cols = ['TODAY', 'MONEYLINE', 'SPREAD', 'TOTAL']

    # Display Game
    game_display = game[display_cols]
    game_display.rename(columns={'TODAY':'Team', 'MONEYLINE':'Moneyline', 'SPREAD':'Spread', 'TOTAL':'Total'}, inplace=True)


    # add metrics
    col1, col2 = st.columns(2)
    with col1:
        col1.image('data/team/logos/'+team1+'.png', width=200)
        st.subheader('**' + team1 + '**')
        st.write('**Moneyline:** ' + str(game['MONEYLINE'].iloc[0]))
        st.write('**Spread:** ' + str(game['SPREAD'].iloc[0]))
        st.write("")
    
    with col2:
        st.image('data/team/logos/'+team2+'.png', width = 200)
        st.subheader('**' + team2 + '**')
        st.write('**Moneyline:** ' + str(game['MONEYLINE'].iloc[1]))
        st.write('**Spread:** ' + str(game['SPREAD'].iloc[1]))
        st.write("")

    # Page Break
    st.write("---")

    # Display Team Stats
    st.write('**Team Stats**')
    team_1_boxes = boxes_22[boxes_22['trad_team'] == team1]
    team_2_boxes = boxes_22[boxes_22['trad_team'] == team2]
    league_averages_22 = boxes_22.mean()
    league_averages_22 = pd.DataFrame(league_averages_22).T
    # get season averages
    team_1_avg = team_1_boxes.mean()
    team_1_avg = pd.DataFrame(team_1_avg).T
    team_2_avg = team_2_boxes.mean()
    team_2_avg = pd.DataFrame(team_2_avg).T
    # get last 10 averages
    team_1_last10 = team_1_boxes.head(10).mean()
    team_1_last10 = pd.DataFrame(team_1_last10).T
    team_2_last10 = team_2_boxes.head(10).mean()
    team_2_last10 = pd.DataFrame(team_2_last10).T
    # get last 5 averages
    team_1_last5 = team_1_boxes.head(5).mean()
    team_1_last5 = pd.DataFrame(team_1_last5).T

    team_2_last5 = team_2_boxes.head(5).mean()
    team_2_last5 = pd.DataFrame(team_2_last5).T

    trad_display_columns = ['trad_pts', 'trad_fg%', 'trad_3pa', 'trad_3p%', 'trad_fta', 'trad_tov']
    adv_display_columns = ['adv_offrtg', 'adv_defrtg', 'adv_efg%', 'adv_pace','adv_ts%',
                             'adv_tov%']

    all_display_columns = trad_display_columns + adv_display_columns

    # display averages
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(team1)
        st.write('**Averages**')
        team_1_avg_disp = team_1_avg[all_display_columns]
        # rename columns
        team_1_avg_disp.rename(columns={'trad_pts':'PTS', 'trad_fg%':'FG%', 'trad_3pa':'3PA', 'trad_3p%':'3P%', 'trad_fta':'FTA',
                                        'trad_tov':'TOV', 'adv_offrtg':'OFFRtg', 'adv_defrtg':'DEFRtg',
                                        'adv_efg%':'eFG%', 'adv_pace':'PACE', 'adv_ts%':'TS%', 'adv_tov%':'TOV%'}, inplace=True)
        seaz_avg = team_1_avg_disp
    
        team_1_last10_disp = team_1_last10[all_display_columns]
        # rename columns
        team_1_last10_disp.rename(columns={'trad_pts':'PTS', 'trad_fg%':'FG%', 'trad_3pa':'3PA', 'trad_3p%':'3P%', 'trad_fta':'FTA',
                                        'trad_tov':'TOV', 'adv_offrtg':'OFFRtg', 'adv_defrtg':'DEFRtg',
                                        'adv_efg%':'eFG%', 'adv_pace':'PACE', 'adv_ts%':'TS%', 'adv_tov%':'TOV%'}, inplace=True)
        laz10_avg = team_1_last10_disp

        # Difference between Season Avg and Last 10 Avg

        team_1_diff = team_1_last10 - team_1_avg
        team_1_diff_disp = team_1_diff[all_display_columns]
        # rename columns
        team_1_diff_disp.rename(columns={'trad_pts':'PTS', 'trad_fg%':'FG%', 'trad_3pa':'3PA', 'trad_3p%':'3P%', 'trad_fta':'FTA',
                                        'trad_tov':'TOV', 'adv_offrtg':'OFFRtg', 'adv_defrtg':'DEFRtg',
                                        'adv_efg%':'eFG%', 'adv_pace':'PACE', 'adv_ts%':'TS%', 'adv_tov%':'TOV%'}, inplace=True)
        diffz = team_1_diff_disp

        # merge season avg and last 10 avg into one dataframe
        team_1_avg_disp = pd.concat([team_1_avg_disp, team_1_last10_disp], axis=0)
        team_1_avg_disp = team_1_avg_disp.reset_index(drop=True)
        team_1_avg_disp = team_1_avg_disp.rename(index={0:'Season Avg', 1:'Last 10 Avg'})
        st.table(team_1_avg_disp.style.format("{:.2f}"))

        team_1_boxes_fixed = team_1_boxes[all_display_columns]
        # rename columns
        team_1_boxes_fixed.rename(columns={'trad_pts':'PTS', 'trad_fg%':'FG%', 'trad_3pa':'3PA', 'trad_3p%':'3P%', 'trad_fta':'FTA',
                                        'trad_tov':'TOV', 'adv_offrtg':'OFFRtg', 'adv_defrtg':'DEFRtg',
                                        'adv_efg%':'eFG%', 'adv_pace':'PACE', 'adv_ts%':'TS%', 'adv_tov%':'TOV%'}, inplace=True)
        # fix boxes_22
        boxes_22_fixed = boxes_22.copy()
        # rename columns
        boxes_22_fixed.rename(columns={'trad_pts':'PTS', 'trad_fg%':'FG%', 'trad_3pa':'3PA', 'trad_3p%':'3P%', 'trad_fta':'FTA',
                                        'trad_tov':'TOV', 'adv_offrtg':'OFFRtg', 'adv_defrtg':'DEFRtg',
                                        'adv_efg%':'eFG%', 'adv_pace':'PACE', 'adv_ts%':'TS%', 'adv_tov%':'TOV%'}, inplace=True)
        


        # violin plot each column from team_1_boxes with all_display_columns, but one plot total
        st.write('**Season Distribution:**')
        st.write(f'What stat do you want to see for **{team1}**?')
        stat = st.selectbox('', team_1_boxes_fixed.columns, key = keyyy)
        fig = px.violin(team_1_boxes_fixed, y=stat, box=True, points='all', hover_data=team_1_boxes_fixed.columns)
        # add league violin to the plot, from boxes_22
        fig.add_trace(go.Violin(y=boxes_22_fixed[stat], name='League', box_visible=True, meanline_visible=True))
        fig.update_layout(height=500)
        st.plotly_chart(fig)

        # import gaussian_kde from scipy.stats
        from scipy.stats import gaussian_kde

        # Plot the histograms and kde for team1 and league average on same plot, opacity 0.5
        st.write('**Chosen Statistic, Histogram Comparison with Kernel Density Estimation:**')
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=team_1_boxes_fixed[stat], name=team1, histnorm='probability density', opacity=0.15, ybins = dict(start=0, end=1, size=0.05)))
        fig.add_trace(go.Histogram(x=boxes_22_fixed[stat], name='League', histnorm='probability density', opacity=0.15, ybins = dict(start=0, end=1, size=0.1)))
        fig.update_layout(barmode='overlay')

        # Add a density curve to both histograms
        x1 = team_1_boxes_fixed[stat]
        x2 = boxes_22_fixed[stat]
        x1_dens = gaussian_kde(x1)
        x2_dens = gaussian_kde(x2)
        x1_dens.covariance_factor = lambda : .25
        x1_dens._compute_covariance()
        x2_dens.covariance_factor = lambda : .25
        x2_dens._compute_covariance()
        x1_range = np.linspace(min(x1), max(x1), 100)
        x2_range = np.linspace(min(x2), max(x2), 100)
        fig.add_trace(go.Scatter(x=x1_range, y=x1_dens(x1_range), name='Team 1 Density', line_color='blue'))
        fig.add_trace(go.Scatter(x=x2_range, y=x2_dens(x2_range), name='League Density', line_color='orange'))

        fig.update_layout(height=500)
        st.plotly_chart(fig)

        # # plot adv_pace, adv_offrtg, adv_defrtg, adv_ts% for team_1 in a box plot versus league
        # st.write('**Season Box Plot:**')
        # fig = go.Figure()
        # fig.add_trace(go.Box(y=team_1_boxes['adv_pace'], name='PACE', marker_color = 'lightseagreen' ))
        # fig.add_trace(go.Box(y=boxes_22['adv_pace'], name='League PACE', marker_color = 'indianred'))
        # fig.add_trace(go.Box(y=team_1_boxes['adv_offrtg'], name='OFFRtg', marker_color = 'lightseagreen'))
        # fig.add_trace(go.Box(y=boxes_22['adv_offrtg'], name='League OFFRtg', marker_color = 'indianred'))

        # fig.add_trace(go.Box(y=team_1_boxes['adv_defrtg'], name='DEFRtg', marker_color = 'lightseagreen'))
        # fig.add_trace(go.Box(y=boxes_22['adv_defrtg'], name='League DEFRtg', marker_color = 'indianred'))

        # fig.update_layout(height=500)
        # st.plotly_chart(fig)


    with col2:
        st.subheader(team2)
        st.write('**Averages:**')
        team_2_avg_disp = team_2_avg[all_display_columns]
        # rename columns
        team_2_avg_disp.rename(columns={'trad_pts':'PTS', 'trad_fg%':'FG%', 'trad_3pa':'3PA', 'trad_3p%':'3P%', 'trad_fta':'FTA',
                                        'trad_ft%':'FT%', 'trad_blk':'BLK', 'trad_tov':'TOV', 'adv_offrtg':'OFFRtg', 'adv_defrtg':'DEFRtg',
                                        'adv_efg%':'eFG%', 'adv_pace':'PACE', 'adv_ts%':'TS%', 'adv_tov%':'TOV%'}, inplace=True)
        t1avg = team_2_avg_disp

        team_2_last10_disp = team_2_last10[all_display_columns]
        # rename columns
        team_2_last10_disp.rename(columns={'trad_pts':'PTS', 'trad_fg%':'FG%', 'trad_3pa':'3PA', 'trad_3p%':'3P%', 'trad_fta':'FTA',
                                        'trad_ft%':'FT%', 'trad_blk':'BLK', 'trad_tov':'TOV', 'adv_offrtg':'OFFRtg', 'adv_defrtg':'DEFRtg',
                                        'adv_efg%':'eFG%', 'adv_pace':'PACE', 'adv_ts%':'TS%', 'adv_tov%':'TOV%'}, inplace=True)
        t1last10 = team_2_last10_disp

        # Difference between Season Avg and Last 10 Avg
        team_2_diff = team_2_last10 - team_2_avg
        team_2_diff_disp = team_2_diff[all_display_columns]
        # rename columns
        team_2_diff_disp.rename(columns={'trad_pts':'PTS', 'trad_fg%':'FG%', 'trad_3pa':'3PA', 'trad_3p%':'3P%', 'trad_fta':'FTA',
                                        'trad_ft%':'FT%', 'trad_blk':'BLK', 'trad_tov':'TOV', 'adv_offrtg':'OFFRtg', 'adv_defrtg':'DEFRtg',
                                        'adv_efg%':'eFG%', 'adv_pace':'PACE', 'adv_ts%':'TS%', 'adv_tov%':'TOV%'}, inplace=True)

        t1diff = team_2_diff_disp

        #merge seasib avg and last 10 avg
        t1merge = pd.concat([t1avg, t1last10], axis=0)
        #reset index
        t1merge.reset_index(inplace=True)
        #drop index column
        t1merge.drop(columns='index', inplace=True)
        #rename indexes
        t1merge.rename(index={0:'Season Avg', 1:'Last 10 Avg'}, inplace=True)
        st.table(t1merge.style.format("{:.2f}"))

        # violin plot each column from team_2_boxes with all_display_columns, but one plot total
        st.write('**Season Distribution:**')
        st.write(f'What stat do you want to see for **{team2}**?')
                # fix team_2_boxes to have same columns as team_1_boxes
        team_2_boxes_fixed = team_2_boxes[all_display_columns]
        # rename columns
        team_2_boxes_fixed.rename(columns={'trad_pts':'PTS', 'trad_fg%':'FG%', 'trad_3pa':'3PA', 'trad_3p%':'3P%', 'trad_fta':'FTA',
                                        'trad_ft%':'FT%', 'trad_blk':'BLK', 'trad_tov':'TOV', 'adv_offrtg':'OFFRtg', 'adv_defrtg':'DEFRtg',
                                        'adv_efg%':'eFG%', 'adv_pace':'PACE', 'adv_ts%':'TS%', 'adv_tov%':'TOV%'}, inplace=True)
        

        stat2 = st.selectbox('', team_2_boxes_fixed.columns, key=keyyy + 1)
        
        fig = px.violin(team_2_boxes_fixed, y=stat2, box=True, points='all', hover_data=team_2_boxes_fixed.columns)
        # add league average line from league_averages_22
        fig.add_trace(go.Violin(y=boxes_22_fixed[stat2], name='League', box_visible=True, meanline_visible=True, ))
        # set height of plot
        fig.update_layout(height=500)
        st.plotly_chart(fig)




        # Add Chosen Statistic, Historic Comparison with KDE, same as above, for team 2, in plotly
        st.write('**Chosen Statistic, Histogram Comparison with Kernel Density Estimation:**')
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=team_2_boxes_fixed[stat], name=team2, histnorm='probability density', opacity=0.15, ybins = dict(start=0, end=1, size=0.05)))
        fig.add_trace(go.Histogram(x=boxes_22_fixed[stat], name='League', histnorm='probability density', opacity=0.15, ybins = dict(start=0, end=1, size=0.1)))
        fig.update_layout(barmode='overlay')

        # Add a density curve to both histograms
        x1 = team_2_boxes_fixed[stat]
        x2 = boxes_22_fixed[stat]
        x1_dens = gaussian_kde(x1)
        x2_dens = gaussian_kde(x2)
        x1_dens.covariance_factor = lambda : .25
        x1_dens._compute_covariance()
        x2_dens.covariance_factor = lambda : .25
        x2_dens._compute_covariance()
        x1_range = np.linspace(min(x1), max(x1), 100)
        x2_range = np.linspace(min(x2), max(x2), 100)
        fig.add_trace(go.Scatter(x=x1_range, y=x1_dens(x1_range), name='Team 1 Density', line_color='blue'))
        fig.add_trace(go.Scatter(x=x2_range, y=x2_dens(x2_range), name='League Density', line_color='orange'))


        fig.update_layout(height=500)
        st.plotly_chart(fig)




        # # plot box plot of adv_pace, adv_offrtg, adv_defrtg
        # fig = go.Figure()
        # fig.add_trace(go.Box(y=team_2_boxes['adv_pace'], name='PACE', marker_color = 'lightseagreen'))
        # fig.add_trace(go.Box(y=boxes_22['adv_pace'], name='League PACE', marker_color = 'indianred'))
        # fig.add_trace(go.Box(y=team_2_boxes['adv_offrtg'], name='OFFRtg', marker_color = 'lightseagreen'))
        # fig.add_trace(go.Box(y=boxes_22['adv_offrtg'], name='League OFFRtg', marker_color = 'indianred'))
        # fig.add_trace(go.Box(y=team_2_boxes['adv_defrtg'], name='DEFRtg', marker_color = 'lightseagreen'))
        # fig.add_trace(go.Box(y=boxes_22['adv_defrtg'], name='League DEFRtg', marker_color = 'indianred'))
        # fig.update_layout(height=500)
        # st.plotly_chart(fig)





    # EXPECTED STARTERS
    starters_df = expected_starters[expected_starters['Away_Team'] == game['team_fixed_abv'].iloc[0]]
    starters_df = starters_df[['Away_Status','Away_Team', 'Away_PG', 'Away_SG', 'Away_SF', 'Away_PF', 'Away_C']]
    starters_df.rename(columns={'Away_Status':'Status', 'Away_Team':'Team', 'Away_PG':'PG', 'Away_SG':'SG', 'Away_SF':'SF', 'Away_PF':'PF', 'Away_C':'C'}, inplace=True)
    starters_df['Status'] = starters_df['Status'].str.strip()

    starters_df2 = expected_starters[expected_starters['Home_Team'] == game['team_fixed_abv'].iloc[1]]
    starters_df2 = starters_df2[['Home_Status','Home_Team', 'Home_PG', 'Home_SG', 'Home_SF', 'Home_PF', 'Home_C']]
    starters_df2.rename(columns={'Home_Status':'Status', 'Home_Team':'Team', 'Home_PG':'PG', 'Home_SG':'SG', 'Home_SF':'SF', 'Home_PF':'PF', 'Home_C':'C'}, inplace=True)
    starters_df2['Status'] = starters_df2['Status'].str.strip()

    # add starters_df2 to starters_df
    starters = pd.concat([starters_df, starters_df2])

    starters_disp = starters[['Status', 'Team', 'PG', 'SG', 'SF', 'PF', 'C']]

    # get usage rates
    usage_rates = player_advanced_metrics[['Player', 'USG%']]

    # get expected minutes
    game_expected_min = expected_minutes[expected_minutes['team'] == game['team_name'].iloc[0]]
    game_expected_min2 = expected_minutes[expected_minutes['team'] == game['team_name'].iloc[1]]

    # add usg to game_expected_min
    game_expected_min = pd.merge(game_expected_min, usage_rates, left_on='name', right_on='Player', how='left')
    game_expected_min2 = pd.merge(game_expected_min2, usage_rates, left_on='name', right_on='Player', how='left')

    # add player_points_percentage, player FG_attempt percentage, player FT_attempt percentage
    player_points_percentage = my_usage_metrics[['trad_player', 'player_points_percentage', 'player_field_goal_attepted_percentage', 'player_free_throws_attempted_percentage']]
    # rename columns
    player_points_percentage.rename(columns={'trad_player':'name'}, inplace=True)
    
    # add player_points_percentage to game_expected_min
    game_expected_min = pd.merge(game_expected_min, player_points_percentage, on='name', how='left')
    game_expected_min2 = pd.merge(game_expected_min2, player_points_percentage, on='name', how='left')

    # display expected minutes
    game_expected_min_display = game_expected_min[['name', 'minutes', 'GTD', 'USG%', 'player_points_percentage', 'player_field_goal_attepted_percentage', 'player_free_throws_attempted_percentage']]
    game_expected_min2_display = game_expected_min2[['name', 'minutes', 'GTD', 'USG%', 'player_points_percentage', 'player_field_goal_attepted_percentage', 'player_free_throws_attempted_percentage']]
    
    # rename columns
    game_expected_min_display.rename(columns={'name':'Player', 'minutes':'minutes', 'GTD':'GTD', 'USG%':'Usage Rate', 
                                                'player_points_percentage':'Points %', 'player_field_goal_attepted_percentage':'FGA %', 
                                                'player_free_throws_attempted_percentage':'FTA %'}, inplace=True)

    game_expected_min2_display.rename(columns={'name':'Player', 'minutes':'minutes', 'GTD':'GTD', 'USG%':'Usage Rate', 
                                                'player_points_percentage':'Points %', 'player_field_goal_attepted_percentage':'FGA %', 
                                                'player_free_throws_attempted_percentage':'FTA %'}, inplace=True)

    ############## OUTS AND MAY NOT PLAY ################

    # Team 1 OUT players
    out_1 = game_expected_min[game_expected_min['OUT'] == 1]
    out_1_display = out_1['name'].reset_index(drop=True)
    out_1_list = out_1['name'].to_list()
    out_1_df = pd.DataFrame(out_1_list, columns=['Player'])

    # Team 2 OUT players
    out_2 = game_expected_min2[game_expected_min2['OUT'] == 1]
    out_2_list = out_2['name'].reset_index(drop=True)
    out_2_list = out_2_list.tolist()
    out_2_df = pd.DataFrame(out_2_list, columns=['Player'])

    # Team 1 MNP players
    may_not_play_1 = game_expected_min[game_expected_min['GTD'] == 1]
    # check

    may_not_play_1_players_list = may_not_play_1['name'].tolist()
    mnp1_df = pd.DataFrame(may_not_play_1_players_list, columns=['Player'])

    # May Not Play, Team 2
    may_not_play_2 = game_expected_min2[game_expected_min2['GTD'] == 1]
    may_not_play_2_players_list = may_not_play_2['name'].tolist()
    mnp2_df = pd.DataFrame(may_not_play_2_players_list, columns=['Player'])

    may_not_play_total = pd.concat([may_not_play_1, may_not_play_2])

    def fix_player_names(df):
        df['first_initial_last_name'] = df['Player'].str.split(' ').str[0] + ' ' + df['Player'].str.split(' ').str[-1]
        return df

    out_1_df = pd.merge(out_1_df, player_names, left_on='Player', 
                                                right_on = 'first_initial_last_name',
                                                how='left')
    out_2_df = pd.merge(out_2_df, player_names, left_on='Player', 
                                                right_on = 'first_initial_last_name',
                                                how='left')
    mnp1_df = pd.merge(mnp1_df, player_names, left_on='Player', 
                                                right_on = 'first_initial_last_name',
                                                how='left')
    mnp2_df = pd.merge(mnp2_df, player_names, left_on='Player', 
                                                right_on = 'first_initial_last_name',
                                                how='left')
 
    # new columns "name_final" = Player if Player does not contain a period, otherwise player_name
    def get_fixed_name(df):
        # if player_name is not null, return player_name
        if df['player_name'] is not np.nan:
            return df['player_name']
        else:
            return df['Player']

    out_1_df['name_final'] = out_1_df.apply(get_fixed_name, axis=1)
    out_2_df['name_final'] = out_2_df.apply(get_fixed_name, axis=1)
    mnp1_df['name_final'] = mnp1_df.apply(get_fixed_name, axis=1)
    mnp2_df['name_final'] = mnp2_df.apply(get_fixed_name, axis=1)

            # add usage rates to out_1_df
    out_1_df = pd.merge(out_1_df, usage_rates, left_on='name_final', right_on = 'Player', how='left')
    out_2_df = pd.merge(out_2_df, usage_rates, left_on='name_final', right_on = 'Player', how='left')
    mnp1_df = pd.merge(mnp1_df, usage_rates, left_on='name_final', right_on = 'Player', how='left')
    mnp2_df = pd.merge(mnp2_df, usage_rates, left_on='name_final', right_on = 'Player', how='left')


    # drop columns
    out_1_df = out_1_df.drop(['Player_x', 'player_name', 'first_initial_last_name', 'name_final'], axis=1)
    out_2_df = out_2_df.drop(['Player_x', 'player_name', 'first_initial_last_name', 'name_final'], axis=1)
    mnp1_df = mnp1_df.drop(['Player_x', 'player_name', 'first_initial_last_name', 'name_final'], axis=1)
    mnp2_df = mnp2_df.drop(['Player_x', 'player_name', 'first_initial_last_name', 'name_final'], axis=1)

    # rename columns
    out_1_df = out_1_df.rename(columns={'Player_y': 'Player'})
    out_2_df = out_2_df.rename(columns={'Player_y': 'Player'})
    mnp1_df = mnp1_df.rename(columns={'Player_y': 'Player'})
    mnp2_df = mnp2_df.rename(columns={'Player_y': 'Player'})

    # drop Player NAs
    out_1_df = out_1_df.dropna(subset=['Player'])
    out_2_df = out_2_df.dropna(subset=['Player'])
    

    # show players who may not play, if applicable
    if len(may_not_play_total) > 0:
        st.markdown('Players who **MAY NOT PLAY**:')
        col1, col2 = st.columns(2)
        if may_not_play_1_players_list != '':
            # add df
            col1.table(mnp1_df)
        else:
            col1.write("No players MNP")
        
        if may_not_play_2_players_list != '':
            col2.table(mnp2_df)
        else:
            col2.write("No players MNP")

    # make line break
    st.write("---")

    # show players who are out
    st.markdown('Players who are **OUT**:')
    col1, col2 = st.columns(2)
    # drop duplicate players
    out_1_df = out_1_df.drop_duplicates(subset=['Player'], keep='first')
    out_2_df = out_2_df.drop_duplicates(subset=['Player'], keep='first')
    col1.table(out_1_df)
    col1.write("")
    col2.table(out_2_df)
    col2.write("")

    starters_disp_t = starters_disp.set_index('Team').T

    ##############    STARTERS    ################

    st.markdown("**Starting Lineups** -- status will be 'Confirmed Lineup' if finalized")
    col1, col2 = st.columns(2)
    # define function to color Status green if Confirmed Lineup
    def color_status(val):
        color = 'green' if len(val) > 16 else 'red'
        return f'background-color: {color}'
    
    # drop everything but status
    starters_1_staus = starters_df[['Status']]
    starters_2_staus = starters_df2[['Status']]

    col1.dataframe(starters_1_staus.style.applymap(color_status, subset=['Status']))
    col2.dataframe(starters_2_staus.style.applymap(color_status, subset=['Status']))

    starters_df_t = starters_df.T
    # drop first row
    starters_df_t = starters_df_t.drop(starters_df_t.index[0])
    # assign column name to 'Starting Lineup'
    old_name = starters_df_t.columns[0]
    new_name = 'Starting Lineup'
    starters_df_t = starters_df_t.rename(columns={old_name: new_name})

    starters_df2_t = starters_df2.T
    # drop first row
    starters_df2_t = starters_df2_t.drop(starters_df2_t.index[0])
    # rename column
    old_name2 = starters_df2_t.columns[0]
    new_name2 = 'Starting Lineup'
    starters_df2_t = starters_df2_t.rename(columns={old_name2: new_name2})


    col1, col2 = st.columns(2)
    col1.dataframe(starters_df_t)
    col2.dataframe(starters_df2_t)

    st.markdown("Team Defense by Position")
    col1, col2 = st.columns(2)
    
    def color_fg(val):
        color = 'green' if int(val) < 10 else 'white'
        return f'background-color: {color}'
    
    team_1_def = d_vs_position[d_vs_position['TEAM'] == team1]
    team_2_def = d_vs_position[d_vs_position['TEAM'] == team2]
    col1.dataframe(team_1_def.style.format(subset=['PTS', 'REB', 'FG%'], formatter="{:.1f}").format(subset=['PTS_rank'], formatter="{:.0f}").applymap(color_fg, subset=['PTS_rank']))
    col2.dataframe(team_2_def.style.format(subset=['PTS', 'REB', 'FG%'], formatter="{:.1f}").format(subset=['PTS_rank'], formatter="{:.0f}").applymap(color_fg, subset=['PTS_rank']))

    st.markdown("Implied Player Production Increase")
    col1, col2 = st.columns(2)
    
    team_1_bad_def = team_1_def[team_1_def['PTS_rank'] < 10]
    team_2_bad_def = team_2_def[team_2_def['PTS_rank'] < 10]

    if len(team_1_bad_def) > 0:
        # make team1baddef have index of position
        team_1_bad_def = team_1_bad_def.set_index('POS')
        # join with starting lineup of other team
        team_1_bad_def = team_1_bad_def.join(starters_df2_t.rename(columns={'Starting Lineup': 'Implied Outperformer(s)'}))
        col1.dataframe(team_1_bad_def.style.format(subset=['PTS', 'REB', 'FG%'], 
                                                    formatter="{:.1f}").format(subset=['PTS_rank'], 
                                                    formatter="{:.0f}").applymap(color_fg, subset=['PTS_rank']))
    else:
        col1.markdown("No positions with especially bad defense")

    if len(team_2_bad_def) > 0:
        # make team2baddef have index of position
        team_2_bad_def = team_2_bad_def.set_index('POS')
        # join with starting lineup of other team
        team_2_bad_def2 = team_2_bad_def.join(starters_df_t.rename(columns={'Starting Lineup': 'Implied Outperformer(s)'}))
        col2.dataframe(team_2_bad_def2.style.format(subset=['PTS', 'REB', 'FG%'], 
                                                    formatter="{:.1f}").format(subset=['PTS_rank'], 
                                                    formatter="{:.0f}").applymap(color_fg, subset=['PTS_rank']))

    else:
        # write 'no positions with especially bad defense' in red color
        col2.write("No positions with especially bad defense", color = 'red')
        
    st.markdown("")
    st.markdown("**Expected Minutes and Usage**")
    # button to show expected minutes
    show_expected_min = st.button("Show Expected Minutes Table", key = skey)
    if show_expected_min:
        col1, col2 = st.columns(2)
        col1.dataframe(game_expected_min_display.style.format(subset=['minutes', 'Points %', 'FGA %', 'FTA %'], formatter="{:.2f}"))
        col2.dataframe(game_expected_min2_display.style.format(subset=['minutes', 'Points %', 'FGA %', 'FTA %'], formatter="{:.2f}"))
    else:
        st.markdown("Click button to show expected minutes")

    # plot scatterplot of minutes vs Points %
    st.markdown("**Expected Minutes vs Team Points Percentage**")
    col1, col2 = st.columns(2)
    # scatterplot of minutes vs Points %
    fig = px.scatter(game_expected_min_display, x="minutes", y="Points %", color="Player", hover_name="Player", title="Minutes vs Points %")
    fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
    #fig.update_layout(legend=dict( yanchor="top", y=0.99, xanchor="left", x=0.01))
    col1.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(game_expected_min2_display, x="minutes", y="Points %", color="Player", hover_name="Player", title="Minutes vs Points %")
    fig2.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
    #fig2.update_layout(legend=dict( yanchor="top", y=0.99, xanchor="left", x=0.01))
    col2.plotly_chart(fig2, use_container_width=True)

    # Player GBG for Who Played When
    player_gbg_22 = player_gbg[player_gbg['trad_season'] == 2022]
    # fix trad_game date so it is sortable
    player_gbg_22['trad_game date'] = pd.to_datetime(player_gbg_22['trad_game date'])
    # make JUST date
    player_gbg_22['trad_game date'] = player_gbg_22['trad_game date'].dt.date
    
    team_1_pgbg = player_gbg_22[player_gbg_22['trad_team'] == team1]
    team_2_pgbg = player_gbg_22[player_gbg_22['trad_team'] == team2]
    # groupby game and player minutes
    team_1_pgbg = team_1_pgbg.groupby(['trad_game date', 'trad_player'])['trad_min'].sum().reset_index()
    team_2_pgbg = team_2_pgbg.groupby(['trad_game date', 'trad_player'])['trad_min'].sum().reset_index()
    # 

    team_1_pgbg = team_1_pgbg.pivot_table(index = 'trad_player', columns = 'trad_game date', values = 'trad_min', aggfunc = 'sum').reset_index()
    # after trad_player, switch the columns the other way around
    team_2_pgbg = team_2_pgbg.pivot_table(index = 'trad_player', columns = 'trad_game date', values = 'trad_min', aggfunc = 'sum').reset_index()

    
    # trad_game date to datetime.date
    # team_1_pgbg['trad_game date'] = pd.to_datetime(team_1_pgbg['trad_game date'])
    # team_1_pgbg['trad_game date'] = team_1_pgbg['trad_game date'].dt.date
    #team_1_pgbg = team_1_pgbg.sort_values(by = 'trad_game date', axis = 1, ascending = False)
    # sort by most recent game
    #team_1_pgbg['trad_game date'] = pd.to_datetime(team_1_pgbg['trad_game date'])
    #team_1_pgbg = team_1_pgbg.sort_values(by = 'trad_game date', ascending = False)

    
    st.markdown("**Who Played When**")
    # add button to show who played when
    show_who_played_when = st.button("Show Who Played When Table", key = wkey)
    if show_who_played_when:
        col1, col2 = st.columns(2)
        t1_cols = team_1_pgbg.columns
        # drop trad_player column
        t1_cols = t1_cols.drop('trad_player')
        # sort columns by most recent game
        t1_cols = t1_cols.sort_values(ascending = True)
        # switch columns from left to right
        t1_cols = t1_cols[::-1]
        tot_cols_1 = ['trad_player'] + list(t1_cols)
        # put back in team_1_pgbg
        team_1_pgbg = team_1_pgbg[tot_cols_1]
        # do the same for team 2
        t2_cols = team_2_pgbg.columns
        t2_cols = t2_cols.drop('trad_player')
        t2_cols = t2_cols.sort_values(ascending = True)
        t2_cols = t2_cols[::-1]
        tot_cols_2 = ['trad_player'] + list(t2_cols)
        team_2_pgbg = team_2_pgbg[tot_cols_2]

        # colors
        def color_fg2(val):
            val = str(val)
            # color is red if val contains nan, else white
            color = 'red' if val == 'nan' else 'white'
            return f'background-color: {color}'

        t2_cols = team_2_pgbg.columns
        t2_cols = t2_cols.drop('trad_player')

        # sort players by team points percentage
        team_1_pgbg = team_1_pgbg.sort_values(by = t1_cols[0], ascending = False)
        team_2_pgbg = team_2_pgbg.sort_values(by = t2_cols[0], ascending = False)

        col1.table(team_1_pgbg.style.format(subset=t1_cols, formatter="{:.0f}").applymap(color_fg2, subset=t1_cols))
        col2.table(team_2_pgbg.style.format(subset=t2_cols, formatter="{:.0f}").applymap(color_fg2, subset=t2_cols))
        st.markdown("""---""")
    else:
        pass

    game_fin = games_df.iloc[a:b]
    st.markdown('**Game Prediction:**')
    # Whichever team has the higher win probability is the predicted winner
    # find the index of the higher win probability
    winner_index = game_fin['Win Probability'].idxmax()
    # find the team name of the higher win probability
    winner = game_fin['Team'][winner_index]
    winner_prob = game_fin['Win Probability'][winner_index]
    st.markdown(f"**{winner} with a win probability of {winner_prob}**")

    game_fin = game_fin[['Team', 'Spread', 'Total', 'Win Probability']]
    st.table(game_fin.style.format(subset=['Spread', 'Total',], formatter="{:.1f}").format(subset= 'Win Probability', formatter="{:.0%}".format))



    a = a + 2
    b = b + 2
    n = n + 1
    keyyy = keyyy + 2
    skey = skey + 2
    wkey = wkey + 2
    

    st.markdown("""---""") 

# st.sidebar.subheader("Trav's Links")
# st.sidebar.markdown("[Lineups](https://www.rotowire.com/basketball/nba-lineups.php/)")
# st.sidebar.markdown("[On/Off](https://www.rotowire.com/basketball/court-on-off.php?team=BKN)")
# st.sidebar.markdown("[Teams](https://sports.yahoo.com/nba/teams/)")
# st.sidebar.markdown("[Odds](https://sportsbook.draftkings.com/leagues/basketball/nba)")

