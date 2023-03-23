import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC   
import datetime
from scipy.stats import norm
import os
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from selenium.common.exceptions import WebDriverException
import plotly as py
import plotly.graph_objs as go
import plotly.express as px
import sklearn as sk
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
import plotly.figure_factory as ff

st.set_page_config(page_title='Player Shooting Tool', page_icon=None, layout="wide", initial_sidebar_state="auto" )

# get current time in pst
pst = datetime.timezone(datetime.timedelta(hours=-8))
# to datetime
pst = datetime.datetime.now(pst)

today = pst.strftime('%Y-%m-%d')

# Load Data

player_numbers = pd.read_csv('data/player/nba_com_info/players_and_photo_links.csv')
# add capitalized player name
player_numbers['Player'] = player_numbers['player_name'].str.title()

# Load Sizes
df_sizes = pd.read_csv('data/player/aggregates_of_aggregates/New_Sizes_and_Positions.csv')

# load game by game data
gbg_df = pd.read_csv('data/player/aggregates/Trad&Adv_box_scores_GameView.csv')

# Load tracking and other data
catch_shoot = pd.read_csv('data/player/nba_com_playerdata/tracking/catch_shoot_' + today + '_.csv')
defensive_impact = pd.read_csv('data/player/nba_com_playerdata/tracking/defensive_impact_' + today + '_.csv')
drives = pd.read_csv('data/player/nba_com_playerdata/tracking/drives_' + today + '_.csv')
elbow_touches = pd.read_csv('data/player/nba_com_playerdata/tracking/elbow_touches_' + today + '_.csv')
paint_touches = pd.read_csv('data/player/nba_com_playerdata/tracking/paint_touches_' + today + '_.csv')
passing = pd.read_csv('data/player/nba_com_playerdata/tracking/passing_' + today + '_.csv')
pull_up_shooting = pd.read_csv('data/player/nba_com_playerdata/tracking/pull_up_shots_' + today + '_.csv')
rebounding = pd.read_csv('data/player/nba_com_playerdata/tracking/rebounds_' + today + '_.csv')
speed_distance = pd.read_csv('data/player/nba_com_playerdata/tracking/speed_distance_' + today + '_.csv')
touches = pd.read_csv('data/player/nba_com_playerdata/tracking/touches_' + today + '_.csv')
shooting_efficiency = pd.read_csv('data/player/nba_com_playerdata/tracking/shooting_efficiency_' + today + '_.csv')

# Load Shooting Data
catch_shoot_shooting = pd.read_csv('data/player/nba_com_playerdata/shooting/catch_and_shoot_' + today + '_.csv')
opp_shooting_5ft = pd.read_csv('data/player/nba_com_playerdata/shooting/opp_shooting_5ft_' + today + '_.csv')
opp_shooting_by_zone = pd.read_csv('data/player/nba_com_playerdata/shooting/opp_shooting_by_zone_' + today + '_.csv')
pullups = pd.read_csv('data/player/nba_com_playerdata/shooting/pullups_' + today + '_.csv')
shooting_splits_5ft = pd.read_csv('data/player/nba_com_playerdata/shooting/shooting_splits_5ft_' + today + '_.csv')
shooting_splits_by_zone = pd.read_csv('data/player/nba_com_playerdata/shooting/shooting_splits_by_zone_' + today + '_.csv')
shot_dash_general = pd.read_csv('data/player/nba_com_playerdata/shooting/shot_dash_general_' + today + '_.csv')

# load game by game data
gbg_df = pd.read_csv('data/player/aggregates/Trad&Adv_box_scores_GameView.csv')

# select team
teams = gbg_df['trad_team'].unique()

# sort teams
teams = np.sort(teams)

# 2022-23 season only
gbg_22 = gbg_df[gbg_df['adv_season'] == 2022]

# read team number from session state
team_number = st.session_state.team_num

# make int
team_number = int(team_number)

# select team box
st.session_state.team = st.sidebar.selectbox('Select Team', teams, index = team_number)

# reassign team to correct
team = st.session_state.team

# get list of players on team
players_22 = gbg_22[gbg_22['trad_team'] == team]['trad_player'].unique()

# sort players
players_22 = np.sort(players_22)

# get player and player_num from session state
player = st.session_state.player
player_num = st.session_state.player_number

#player num to int
player_num = int(player_num)

st.session_state.player = st.sidebar.selectbox('Select Player', players_22 , index = player_num)


player_nba_id = player_numbers[player_numbers['Player'] == player]['nba_id'].iloc[0]

# st.sidebar.write('Player NBA_id: ' + str(player_nba_id))

player_photo = 'data/player/photos/photos/' + str(player_nba_id) + '.png'
# add player photo to sidebar
st.sidebar.image(player_photo, width = 200)


# select position
position_options = ['PG', 'SG', 'SF', 'PF', 'C']
position_index = st.session_state['position_index']
# st.sidebar.write('Position Index: ' + str(position_index))
# make int
position_index_dict = {'PG': 0, 'SG': 1, 'SF': 2, 'PF': 3, 'C': 4}
st.session_state.position = st.sidebar.selectbox('Select Position to evaluate the player at', options = position_options, index = position_index)

position = st.session_state.position

# load player data
player_gbg = gbg_df[gbg_df['trad_player'] == player]
unnamed_cols = [col for col in player_gbg.columns if 'Unnamed' in col or 'unnamed' in col]
player_gbg = player_gbg.drop(columns = unnamed_cols)

# load player size data
player_size = df_sizes[df_sizes['player'] == player]
keepcols = ['player', 'height_final', 'wingspan_final', 'primary_position_bbref']

# calculate median height and wingspan for position with df_sizes in last 5 years
df_sizes = df_sizes[df_sizes['position_season'] >= 2017]

median_height = df_sizes[df_sizes['primary_position_bbref'] == position]['height_final'].median()
median_wingspan = df_sizes[df_sizes['primary_position_bbref'] == position]['wingspan_final'].median()

# identify player height and wingspan
player_size = player_size[keepcols]
player_height = player_size['height_final'].iloc[0]
player_wingspan = player_size['wingspan_final'].iloc[0]

# make a df of just players and their primary position
primary_positions = df_sizes[['player', 'primary_position_bbref', 'position_season']]

# calculate player height percentile using df_sizes and primary position
# drop any heights that are 0
df_sizes = df_sizes[df_sizes['height_final'] > 0]

# get position df
positional_df = df_sizes[df_sizes['primary_position_bbref'] == position]


################ END OF INTRO CODE ####################



# ADD POSITIONS
player_avgs_22['position'] = player_avgs_22['trad_player'].map(primary_positions.set_index('player')['primary_position_bbref'])

# get just players position
position_avgs_22 = player_avgs_22[player_avgs_22['position'] == position]


# add position column to shooting_efficiency
#shooting_efficiency['position'] = shooting_efficiency['PLAYER'].map(primary_positions.set_index('player')['primary_position_bbref'])

# position_avgs_22 = player_avgs_22[player_avgs_22['trad_pos'] == position]

st.subheader('Player Clustering for Similar Players')
st.write(f'''The following chart shows the clusters of players who are similar to {player} based on their advanced and traditional stats.
         \n These stats included: 
         \n - Points Per Minute (PPM), 
         \n - Points, 
         \n - 3P%, 
         \n - Assists, 
         \n - Rebounds, 
         \n - Steals, 
         \n - Effective Field Goal Percentage (eFG%), 
         \n - True Shooting Percentage (TS%), 
         \n - Usage, 
         \n - Offensive Rating, and 
         \n - Defensive Rating.
         \n The clusters are based on the elbow method, which typically shows (in my experience) the optimal number of clusters being four. 
         \n You can change the number of clusters to see how the clusters change. Change the value to 20 to see the most-detailed clusters.
         ''')

# Select columns to use for cluster analysis
cluster_cols = ['ppm', 'trad_pts', 'trad_3p%', 'trad_ast', 'trad_reb', 'trad_stl', 'adv_efg%', 'adv_ts%', 'adv_usg%', 'adv_offrtg', 'adv_defrtg']

from sklearn.cluster import KMeans

# cluster
kmeans = KMeans(n_clusters = 5, random_state = 0).fit(position_avgs_22[cluster_cols])
# add cluster column to player averages
position_avgs_22['cluster'] = kmeans.labels_

# elbow method
# create empty list to store wcss
wcss = []
# loop through 1 to 10 clusters
for i in range(1, 20):
    # fit kmeans
    kmeans = KMeans(n_clusters = i, random_state = 0).fit(position_avgs_22[cluster_cols])
    # append wcss to list
    wcss.append(kmeans.inertia_)
# plot wcss
fig = px.line(x = range(1, 20), y = wcss, title = 'Elbow Method')
fig.update_layout(xaxis_title = 'Number of Clusters', yaxis_title = 'WCSS')
st.plotly_chart(fig, use_container_width = True)

# what is WCSS? -> Within Cluster Sum of Squares
# the sum of the squared distance between each member of the cluster and its centroid
# the smaller the WCSS, the denser the cluster

# Choose cluster number
cluster_num = st.selectbox('Choose Cluster Number based on chart elbow (typically 4)', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40], index = 3)

kmeans_fin = KMeans(n_clusters = cluster_num, random_state = 0).fit(position_avgs_22[cluster_cols])
# add cluster column to player averages
position_avgs_22['cluster'] = kmeans_fin.labels_
# Chart clusters with plotly express in a scatter plot
fig = px.scatter(position_avgs_22, x = 'ppm', y = 'trad_pts', color = 'cluster', title = 'Player Clusters', hover_data = position_avgs_22.columns)
fig.update_layout(xaxis_title = 'Points Per Minute', yaxis_title = 'Points')
st.plotly_chart(fig, use_container_width = True)

# show list of players in same cluster as selected player
st.write('Players in Same Cluster')
cluster_mates = position_avgs_22[position_avgs_22['cluster'] == position_avgs_22[position_avgs_22['trad_player'] == player]['cluster'].values[0]]
# everything from third column on to numeric
num_colz = cluster_mates.columns[2:]
cluster_mates[num_colz] = cluster_mates[num_colz].apply(pd.to_numeric, errors = 'coerce')
# drop unnamed cols
unnamed = [col for col in cluster_mates.columns if 'Unnamed' in col]
cluster_mates = cluster_mates.drop(unnamed, axis = 1)
unnamed2 = [col for col in cluster_mates.columns if 'unnamed' in col]
cluster_mates = cluster_mates.drop(unnamed2, axis = 1)

# rename columns, anything with trad_ remove trad_
cluster_mates.columns = [col.replace('trad_', '') for col in cluster_mates.columns]
# drop adv_season
cluster_mates = cluster_mates.drop(['adv_season', 'season', 'adv_min', 'position'], axis = 1)

st.table(cluster_mates.style.format('{:.1f}', subset = cluster_mates.columns[1:]))


# iterate through clusters until only two players remain
for i in range(1, 50):
    kmeans_find_two = KMeans(n_clusters = i, random_state = 0).fit(position_avgs_22[cluster_cols])
    # add cluster column to player averages
    position_avgs_22['cluster'] = kmeans_find_two.labels_
    # show list of players in same cluster as selected player
    cluster_mates = position_avgs_22[position_avgs_22['cluster'] == position_avgs_22[position_avgs_22['trad_player'] == player]['cluster'].values[0]]
    # get length of cluster_mates
    cluster_mates_len = len(cluster_mates)
    # if cluster_mates_len is 2, break
    if cluster_mates_len == 2:
        break

st.subheader('Most Similar Player:')
st.write('The player you selected is in a final cluster with the following player:')

#fix columns in cluster_mates
# drop unnamed
unnamed = [col for col in cluster_mates.columns if 'Unnamed' in col]
cluster_mates = cluster_mates.drop(unnamed, axis = 1)
unnamed2 = [col for col in cluster_mates.columns if 'unnamed' in col]
cluster_mates = cluster_mates.drop(unnamed2, axis = 1)
# rename columns, anything with trad_ remove trad_
cluster_mates.columns = [col.replace('trad_', '') for col in cluster_mates.columns]
# drop adv_season
cluster_mates = cluster_mates.drop(['adv_season', 'season', 'adv_min', 'position'], axis = 1)


st.table(cluster_mates.style.format('{:.1f}', subset = cluster_mates.columns[1:]))
