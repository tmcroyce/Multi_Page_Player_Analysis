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
import unidecode
import re


st.set_page_config(page_title='Player Shooting Tool', page_icon=None, layout="wide", initial_sidebar_state="auto" )


# Name Cleaning Function
def clean_name(n):
    # Remove any extra spaces
    name = n
    name = " ".join(name.split())

    # Convert to lowercase
    name = name.lower()

    # if 'ii' or 'iii' in name, remove it
    if ' ii' in name:
        name = name.replace(' ii', '')
    if ' iii' in name:
        name = name.replace(' iii', '')

    # Remove international characters
    name = unidecode.unidecode(name)

    # Remove periods
    name = name.replace(".", "")

    # Remove 'jr' or 'sr' from the name
    name_parts = name.split()
    name_parts = [part for part in name_parts if part not in ['jr', 'sr']]
    name = " ".join(name_parts)

    # Capitalize each part of the name
    name = name.title()

    # Special case: names like "McGregor"
    name_parts = re.split(' |-', name)
    name_parts = [part[:2] + part[2:].capitalize() if part.startswith("Mc") else part for part in name_parts]
    name = " ".join(name_parts)

    return name

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

# Fix names in loaded data
gbg_df['trad_player'] = gbg_df['trad_player'].apply(clean_name)
gbg_df['adv_player'] = gbg_df['adv_player'].apply(clean_name)
catch_shoot['PLAYER'] = catch_shoot['PLAYER'].apply(clean_name)
defensive_impact['Player'] = defensive_impact['Player'].apply(clean_name)
drives['PLAYER'] = drives['PLAYER'].apply(clean_name)
elbow_touches['PLAYER'] = elbow_touches['PLAYER'].apply(clean_name)
paint_touches['PLAYER'] = paint_touches['PLAYER'].apply(clean_name)
passing['PLAYER'] = passing['PLAYER'].apply(clean_name)
pull_up_shooting['PLAYER'] = pull_up_shooting['PLAYER'].apply(clean_name)
rebounding['PLAYER'] = rebounding['PLAYER'].apply(clean_name)
speed_distance['PLAYER'] = speed_distance['PLAYER'].apply(clean_name)
touches['Player'] = touches['Player'].apply(clean_name)
shooting_efficiency['PLAYER'] = shooting_efficiency['PLAYER'].apply(clean_name)
catch_shoot_shooting['PLAYER'] = catch_shoot_shooting['PLAYER'].apply(clean_name)
opp_shooting_5ft['Player'] = opp_shooting_5ft['Player'].apply(clean_name)
opp_shooting_by_zone['Player'] = opp_shooting_by_zone['Player'].apply(clean_name)
pullups['PLAYER'] = pullups['PLAYER'].apply(clean_name)
shooting_splits_5ft['Player'] = shooting_splits_5ft['Player'].apply(clean_name)
shooting_splits_by_zone['Player'] = shooting_splits_by_zone['Player'].apply(clean_name)
shot_dash_general['PLAYER'] = shot_dash_general['PLAYER'].apply(clean_name)



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


st.subheader('Player Shooting Metrics')
# 3 columns
col1, col2, col3 = st.columns(3)


col1.write('Shooting Efficiency')

player_shooting_efficiency_init = shooting_efficiency[shooting_efficiency['PLAYER'] == player]
# transpose the dataframe
player_shooting_efficiency = player_shooting_efficiency_init.T
# drop first 8 rows
player_shooting_efficiency = player_shooting_efficiency.iloc[8:]
# rename columns
player_shooting_efficiency.columns = ['Shooting Efficiency Metrics']

primary_positions = primary_positions[primary_positions['position_season'] == 2022]


# add position column to shooting_efficiency
shooting_efficiency['position'] = shooting_efficiency['PLAYER'].map(primary_positions.set_index('player')['primary_position_bbref'])

# get averages for player position
position_shooting_efficiency = shooting_efficiency[shooting_efficiency['position'] == position]
position_shooting_efficiency_mean = position_shooting_efficiency.groupby('position').mean().reset_index()

# transpose, drop first 7 rows, add to player_shooting_efficiency
position_shooting_efficiency_mean = position_shooting_efficiency_mean.T
position_shooting_efficiency_mean = position_shooting_efficiency_mean.iloc[7:]
position_shooting_efficiency_mean.columns = ['Position Average']
player_shooting_efficiency2 = pd.concat([player_shooting_efficiency, position_shooting_efficiency_mean], axis = 1)

# if player is not in position_shooting_efficiency, add them
if player not in position_shooting_efficiency['PLAYER'].values:
    # add player_shooting_efficiency_init to position_shooting_efficiency
    position_shooting_efficiency = position_shooting_efficiency.append(player_shooting_efficiency_init)

# add percentile columns to position_shooting_efficiency
for col in position_shooting_efficiency.columns:
    if col != 'PLAYER' and col != 'position':
        position_shooting_efficiency[col + '_percentile'] = position_shooting_efficiency[col].rank(pct = True)

# get player percentile
player_percentile = position_shooting_efficiency[position_shooting_efficiency['PLAYER'] == player]

# add player percentile to player_shooting_efficiency2
player_percentile = player_percentile.T
player_percentile.columns = ['Player Percentile']
# drop first 29 rows
player_percentile = player_percentile.iloc[29:]
# rename all indexes, replacing '_percentile' with ''
player_percentile.index = [col.replace('_percentile', '') for col in player_percentile.index]

player_shooting_efficiency2 = pd.concat([player_shooting_efficiency2, player_percentile], axis = 1)

# function to color code Player Percentiles
def color_code_percentile(val):
    if val < 0.4:
        color = 'red'
    elif val < 0.45 and val > 0.4:
        color = 'orange'
    elif val < 0.55 and val > 0.45:
        color = 'white',
    elif val > 0.55 and val < 0.75:
        color = 'lightgreen'
    elif val > 0.75:
        color = 'green'
    else:
        color = 'white'
    # return highlight color (background)
    return 'background-color: %s' % color

#Test -- Radar chart of player shooting efficiencies percentiles
efficiency_rows = [row for row in player_shooting_efficiency2.index if '%' in row ]
volume_rows = [row for row in player_shooting_efficiency2.index if '%' not in row ]

col2.subheader('Efficiency Metrics')

# radar chart efficiency rows player percentile
fig = go.Figure()

radar_fill_color = 'rgba(34, 97, 153, 0.6)'
fig.add_trace(go.Scatterpolar(
        r=player_shooting_efficiency2.loc[efficiency_rows, 'Player Percentile'],
        theta=efficiency_rows,
        fill='toself',
        fillcolor=radar_fill_color,
        name='Player Percentile'
))


fig.update_layout(
        polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
        showlegend=True
)
# Increase text size
fig.update_layout(font_size=16)



col2.plotly_chart(fig)

# radar chart volume rows player percentile
col3.subheader('Volume / Scoring Metrics')
fig = go.Figure()

fig.add_trace(go.Scatterpolar(
        r=player_shooting_efficiency2.loc[volume_rows, 'Player Percentile'],
        theta=volume_rows,
        fill='toself',
        fillcolor=radar_fill_color,
        name='Player Percentile'
))


fig.update_layout(
        polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
        showlegend=True
)
# Increase text size
fig.update_layout(font_size=16)

col3.plotly_chart(fig)





col1.table(player_shooting_efficiency2.style.format('{:.2f}').applymap(color_code_percentile, subset = ['Player Percentile']))

#############################################################################################################

col1, col2, col3 = st.columns(3)
st.write('---')
col1.subheader('Player Shooting by Zone')

def shooting_by_zone():

    # shooting_splits_by_zone
    player_shooting_splits_by_zone_init = shooting_splits_by_zone[shooting_splits_by_zone['Player'] == player]

    # assign final columns 
    shooting_by_zone_final_cols = ['Restricted Area_FGA', 'Restricted Area_FG%', 'In The Paint (Non-RA)_FGA', 
                                    'In The Paint (Non-RA)_FG%', 'Mid-Range_FGA', 'Mid-Range_FG%', 'Left Corner 3._FGA', 
                                    'Left Corner 3._FG%', 'Right Corner 3._FGA', 'Right Corner 3._FG%', 'Above the Break 3._FGA', 
                                    'Above the Break 3._FG%']

    # assign final columns to player_shooting_splits_by_zone_init
    player_shooting_splits_by_zone_init = player_shooting_splits_by_zone_init

    # transpose the dataframe
    player_shooting_splits_by_zone = player_shooting_splits_by_zone_init.T

    # add position column to shooting_splits_by_zone
    shooting_splits_by_zone['position'] = shooting_splits_by_zone['Player'].map(primary_positions.set_index('player')['primary_position_bbref'])

    # get averages for player POSITION
    position_shooting_splits_by_zone_init = shooting_splits_by_zone[shooting_splits_by_zone['position'] == position]

    # change '-' values to 0
    position_shooting_splits_by_zone_init = position_shooting_splits_by_zone_init.replace('-', 0)

    # assign final columns to position_shooting_splits_by_zone
    position_shooting_splits_by_zone = position_shooting_splits_by_zone_init[shooting_by_zone_final_cols]

    # make all values numeric
    position_shooting_splits_by_zone = position_shooting_splits_by_zone.apply(pd.to_numeric, errors = 'coerce')
    # make mean df
    position_shooting_splits_by_zone_mean = position_shooting_splits_by_zone.mean().reset_index()


    # add position average to player_shooting_splits_by_zone using left join
    player_shooting_splits_by_zone = player_shooting_splits_by_zone.merge(position_shooting_splits_by_zone_mean, 
                                        how = 'left', left_index = True, right_on = 'index')

    # reset index to index
    player_shooting_splits_by_zone = player_shooting_splits_by_zone.set_index('index')

    # Add Player Percentile
    # if player is not in position_shooting_splits_by_zone, add them
    if player not in position_shooting_splits_by_zone_init['Player'].values:
        # add player to position_shooting_splits_by_zone
        position_shooting_splits_by_zone_init = position_shooting_splits_by_zone_init.append(player_shooting_splits_by_zone_init)

    # drop unnamed cols
    unnamed = [col for col in position_shooting_splits_by_zone_init.columns if 'Unnamed' in col]
    position_shooting_splits_by_zone_init = position_shooting_splits_by_zone_init.drop(unnamed, axis = 1)
    # drop team and position cols
    position_shooting_splits_by_zone_init = position_shooting_splits_by_zone_init.drop(['Team', 'position'], axis = 1)
    # ADD percentile columns
    # final cols plus player
    final_cols_and_player = ['Player'] + shooting_by_zone_final_cols 

    position_shooting_splits_by_zone_init = position_shooting_splits_by_zone_init[final_cols_and_player]

    # make sure all values (other than index) are numeric
    position_shooting_splits_by_zone_init[shooting_by_zone_final_cols] = position_shooting_splits_by_zone_init[shooting_by_zone_final_cols].apply(pd.to_numeric, errors = 'coerce')

    for col in position_shooting_splits_by_zone_init.columns:
        # get percentile
        position_shooting_splits_by_zone_init[col + '_percentile'] = position_shooting_splits_by_zone_init[col].rank(pct = True)
    # drop nan in player col
    position_shooting_splits_by_zone_init = position_shooting_splits_by_zone_init.dropna(subset = ['Player'])


    # get player percentile
    player_percentile = position_shooting_splits_by_zone_init[position_shooting_splits_by_zone_init['Player'] == player]
    # drop all columns except percentile columns
    player_percentile = player_percentile[[col for col in player_percentile.columns if 'percentile' in col]]

    player_percentile_t = player_percentile.T

    # drop any rows with nan
    player_shooting_splits_by_zone = player_shooting_splits_by_zone.dropna()

    # rename indexes, removing _percentile
    player_percentile_t.index = [col.replace('_percentile', '') for col in player_percentile_t.index]


    # add player percentile to player_shooting_splits_by_zone
    player_shooting_splits_by_zone = player_shooting_splits_by_zone.merge(player_percentile_t, how = 'left', left_index = True, right_index = True)

    # rename columns
    player_shooting_splits_by_zone.columns = ['Player', 'Position Average', 'Player Percentile']
    # make sure Player column is all numeric
    player_shooting_splits_by_zone['Player'] = player_shooting_splits_by_zone['Player'].apply(pd.to_numeric, errors = 'coerce').round(2)

    # make sure position average column is all numeric
    player_shooting_splits_by_zone['Position Average'] = player_shooting_splits_by_zone['Position Average'].apply(pd.to_numeric, errors = 'coerce').round(2)

    # show df
    return player_shooting_splits_by_zone


player_shooting_splits_by_zone = shooting_by_zone()
col1.table(player_shooting_splits_by_zone.style.format('{:.2f}').applymap(color_code_percentile, subset = ['Player Percentile']))

# efficiency rows by zone
efficiency_rows_zone = [row for row in player_shooting_splits_by_zone.index if "%" in row]
# volume rows by zone
volume_rows_zone = [row for row in player_shooting_splits_by_zone.index if "%" not in row]

# radar chart efficiency by zone
fig = go.Figure()

fig.add_trace(go.Scatterpolar(
        r = player_shooting_splits_by_zone.loc[efficiency_rows_zone, 'Player Percentile'],
        theta = efficiency_rows_zone,
        fill = 'toself',
        fillcolor= radar_fill_color,
        name = 'Player'
    ))


fig.update_layout(
    polar = dict(
        radialaxis = dict(
            visible = True,
            range = [0, 1]
        )),
    showlegend = False
)

fig.update_layout(font_size = 16)

col2.plotly_chart(fig, use_container_width = True)

# radar chart volume by zone
fig = go.Figure()

fig.add_trace(go.Scatterpolar(
        r = player_shooting_splits_by_zone.loc[volume_rows_zone, 'Player Percentile'],
        theta = volume_rows_zone,
        fill = 'toself',
        fillcolor= radar_fill_color,
        name = 'Player'
    ))

fig.update_layout(
    polar = dict(
        radialaxis = dict(
            visible = True,
            range = [0, 1]
        )),
    showlegend = False
)

# larger text
fig.update_layout(font_size = 16)

col3.plotly_chart(fig, use_container_width = True)




####################################################################################################################

col1, col2, col3 = st.columns(3)
col1.subheader('Shooting by Distance (5ft)')
st.write('---')

def shooting_by_distance():

    # shooting_splits_5ft
    player_shooting_splits_5ft_init = shooting_splits_5ft[shooting_splits_5ft['Player'] == player]
    # drop fgm cols
    fgm_cols = [col for col in player_shooting_splits_5ft_init.columns if 'FGM' in col]
    player_shooting_splits_5ft = player_shooting_splits_5ft_init.drop(columns = fgm_cols)

    # add transposed version of player_shooting_splits_5ft
    player_shooting_splits_5ft_t = player_shooting_splits_5ft.T

    # rename column to 'player metrics'
    player_shooting_splits_5ft_t.columns = ['Player Metrics']

    # add position column to shooting_splits_5ft
    shooting_splits_5ft['position'] = shooting_splits_5ft['Player'].map(primary_positions.set_index('player')['primary_position_bbref'])

    # add position_shooting_splits_5ft 
    position_shooting_splits_5ft = shooting_splits_5ft[shooting_splits_5ft['position'] == position]
    # replace - with 0
    position_shooting_splits_5ft = position_shooting_splits_5ft.replace('-', 0)

    # get columns besides player and position
    cols = [col for col in position_shooting_splits_5ft.columns if col not in ['Player', 'position']]

    #make sure values are numeric
    position_shooting_splits_5ft[cols] = position_shooting_splits_5ft[cols].apply(pd.to_numeric, errors = 'coerce')
    # get mean
    position_shooting_splits_5ft_mean = position_shooting_splits_5ft.mean()


    # rename column in position_shooting_splits_5ft_mean to Position Average
    position_shooting_splits_5ft_mean = position_shooting_splits_5ft_mean.rename('Position Average')


    # add position average to player_shooting_splits_5ft using left join
    player_shooting_splits_5ft = pd.merge(player_shooting_splits_5ft_t, position_shooting_splits_5ft_mean, how = 'left', left_index = True, right_index = True)

    # add player to position_shooting_splits_5ft IF they are not already in it
    if player not in position_shooting_splits_5ft['Player'].values:
        position_shooting_splits_5ft = position_shooting_splits_5ft.append(player_shooting_splits_5ft_init)
        
    # drop unnamed cols
    unnamed_cols = [col for col in position_shooting_splits_5ft.columns if 'Unnamed' in col]
    position_shooting_splits_5ft = position_shooting_splits_5ft.drop(columns = unnamed_cols)

    # drop team and position cols
    position_shooting_splits_5ft = position_shooting_splits_5ft.drop(['Team', 'position'], axis = 1)

    # Make sure all values (other than Player column) are numeric
    position_shooting_splits_5ft[position_shooting_splits_5ft.columns[1:]] = position_shooting_splits_5ft[position_shooting_splits_5ft.columns[1:]].apply(pd.to_numeric, errors = 'coerce')


    # ADD percentile columns
    for col in position_shooting_splits_5ft.columns:
        # get percentile
        position_shooting_splits_5ft[col + '_percentile'] = position_shooting_splits_5ft[col].rank(pct = True)


    # get player percentile
    player_percentile = position_shooting_splits_5ft[position_shooting_splits_5ft['Player'] == player]
    # drop all columns except percentile columns
    player_percentile = player_percentile[[col for col in player_percentile.columns if 'percentile' in col]]

    # transpose player_percentile
    player_percentile_t = player_percentile.T

    # rename indexes, removing _percentile
    player_percentile_t.index = [col.replace('_percentile', '') for col in player_percentile_t.index]

    # rename column to Player Percentile
    player_percentile_t.columns = ['Player Percentile']

    # add player percentile to player_shooting_splits_5ft
    player_shooting_splits_5ft = player_shooting_splits_5ft.merge(player_percentile_t, how = 'left', left_index = True, right_index = True)

    # drop unnamed rows
    unnamed_rows = [row for row in player_shooting_splits_5ft.index if 'Unnamed' in row]
    player_shooting_splits_5ft = player_shooting_splits_5ft.drop(index = unnamed_rows)
    # drop player and team rows
    player_shooting_splits_5ft = player_shooting_splits_5ft.drop(['Player', 'Team'])


    # make sure all values are numeric
    player_shooting_splits_5ft[player_shooting_splits_5ft.columns] = player_shooting_splits_5ft[player_shooting_splits_5ft.columns].apply(pd.to_numeric, errors = 'coerce')


    # show df
    return player_shooting_splits_5ft

player_shooting_splits_5ft = shooting_by_distance()
col1.table(player_shooting_splits_5ft.style.format('{:.2f}').applymap(color_code_percentile, subset = ['Player Percentile']))

efficiency_rows_5ft = [row for row in player_shooting_splits_5ft.index if '%' in row]
volume_rows_5ft = [row for row in player_shooting_splits_5ft.index if '%' not in row]

# plot radar chart
fig = go.Figure()

fig.add_trace(go.Scatterpolar(
        r = player_shooting_splits_5ft.loc[efficiency_rows_5ft, 'Player Percentile'],
        theta = efficiency_rows_5ft,
        fill = 'toself',
        fillcolor = radar_fill_color,
        name = 'Player'
    ))

fig.update_layout(
    polar = dict(
        radialaxis = dict(
            visible = True,
            range = [0, 1]
        )),
    showlegend = False
)

# larger text
fig.update_layout(font_size = 16)

col2.plotly_chart(fig)

# volume
fig = go.Figure()

fig.add_trace(go.Scatterpolar(
        r = player_shooting_splits_5ft.loc[volume_rows_5ft, 'Player Percentile'],
        theta = volume_rows_5ft,
        fill = 'toself',
        fillcolor = radar_fill_color,
        name = 'Player'
    ))

fig.update_layout(
    polar = dict(
        radialaxis = dict(
            visible = True,
            range = [0, 1]
        )),
    showlegend = False
)

# larger text
fig.update_layout(font_size = 16)

col3.plotly_chart(fig)


####################################################################################################################





######### SHOT DASHBOARD #########

st.write('Shot Dashboard (General)')
# shot_dash_general
player_shot_dash_general = shot_dash_general[shot_dash_general['PLAYER'] == player]
# drop fgm cols
fgm_cols = [col for col in player_shot_dash_general.columns if 'FGM' in col]
player_shot_dash_general = player_shot_dash_general.drop(columns = fgm_cols)
#drop unnamed cols
unnamed_cols = [col for col in player_shot_dash_general.columns if 'Unnamed' in col]
player_shot_dash_general = player_shot_dash_general.drop(columns = unnamed_cols)


# drop some cols
cols_drop = ['TEAM', 'AGE', 'GP', 'G', 'FREQ']
player_shot_dash_general = player_shot_dash_general.drop(columns = cols_drop)

# sdg_numcols are all columns after index
sdg_numcols = ['FGA', 'FG%', 'EFG%', '2FG FREQ', '2FGA', '2FG%', '3FG FREQ', '3PM', '3PA', '3P%']
# turn all columns numeric
player_shot_dash_general[sdg_numcols] = player_shot_dash_general[sdg_numcols].apply(pd.to_numeric)

# add positions to shot_dash_general using 'PLAYER' and 'player' columns
shot_dash_general['position'] = shot_dash_general['PLAYER'].map(primary_positions.set_index('player')['primary_position_bbref'])

# get averages for player position
position_shot_dash_general = shot_dash_general[shot_dash_general['position'] == position]

# drop fgm cols -- we can figure that out through % and fga
fgm_cols = [col for col in position_shot_dash_general.columns if 'FGM' in col]
position_shot_dash_general = position_shot_dash_general.drop(columns = fgm_cols)

#drop unnamed cols
unnamed_cols = [col for col in position_shot_dash_general.columns if 'Unnamed' in col]
position_shot_dash_general = position_shot_dash_general.drop(columns = unnamed_cols)

# drop more cols
cols_drop = ['TEAM', 'AGE', 'GP', 'G', 'FREQ']
position_shot_dash_general = position_shot_dash_general.drop(columns = cols_drop)

# replace- with 0
position_shot_dash_general = position_shot_dash_general.replace('-', 0)


# turn numb_cols numeric
position_shot_dash_general[sdg_numcols] = position_shot_dash_general[sdg_numcols].apply(pd.to_numeric, errors = 'coerce')

# get averages, dropping any zero or null values
position_avg_shot_dash_general = position_shot_dash_general[sdg_numcols].astype(float).mean(axis = 0).to_frame().T
position_avg_shot_dash_general['PLAYER'] = 'Position Average'


# concat player_shot_dash_general and position_avg_shot_dash_general
position_shot_dash_general_comp = pd.concat([player_shot_dash_general, position_avg_shot_dash_general], axis = 0)
position_shot_dash_general_comp = position_shot_dash_general_comp.reset_index()


# check if player is in position_shot_dash_general. If not, add player_shot_dash_general to position_shot_dash_general
if player not in position_shot_dash_general['PLAYER'].values:
    position_shot_dash_general = pd.concat([position_shot_dash_general, player_shot_dash_general], axis = 0)

# get player percentile for each column
for col in sdg_numcols:
    position_shot_dash_general[col + '_percentile'] = position_shot_dash_general[col].rank(pct = True, method = 'first')

# multiple by 100
for col in position_shot_dash_general.columns:
    if 'percentile' in col:
        position_shot_dash_general[col] = position_shot_dash_general[col] * 100

# find player row
player_row = position_shot_dash_general[position_shot_dash_general['PLAYER'] == player]

player_percentile_cols = [col for col in player_row.columns if 'percentile' in col]

# cols should be index and player_percentile_cols
cols = ['PLAYER'] + player_percentile_cols
player_row = player_row[cols]

# concat player_row to position_shot_dash_general_comp
# rename player_row cols, getting rid of '_percentile'
player_row = player_row.rename(columns = {col: col.replace('_percentile', '') for col in player_row.columns})
# concat to position_shot_dash_general_comp

# rename player name to 'Position Percentile'
player_row['PLAYER'] = 'Position Percentile'
# drop index
player_row = player_row.reset_index()

position_shot_dash_general_comp = pd.concat([position_shot_dash_general_comp, player_row], axis = 0)
position_shot_dash_general_comp = position_shot_dash_general_comp.reset_index()
# drop index
position_shot_dash_general_comp = position_shot_dash_general_comp.drop(columns = ['index'])
# drop index and level_0, make PLAYER index
position_shot_dash_general_comp = position_shot_dash_general_comp.drop(columns = ['level_0']).set_index('PLAYER')

percentile_cols = [col for col in position_shot_dash_general_comp.columns if '%' in col]

# identify third row
percentile_row = position_shot_dash_general_comp.iloc[2]

no_third_row = position_shot_dash_general_comp.drop(index = percentile_row.name)
st.table(no_third_row.style.format('{:.1f}'))

# only_third_row is dropping the first two
only_third_row = position_shot_dash_general_comp.drop(index = no_third_row.index)

def color_code_percentile2(val):
    if val < 40:
        color = 'red'
    elif val < 45 and val > 40:
        color = 'orange'
    elif val < 55 and val > 45:
        color = 'white',
    elif val > 55 and val < 75:
        color = 'lightgreen'
    elif val > 75:
        color = 'green'
    else:
        color = 'white'
    # return highlight color (background)
    return 'background-color: %s' % color

# add third row
st.table(only_third_row.style.format('{:.1f}').applymap(color_code_percentile2))
import base64



# get player percentile for each column
for col in sdg_numcols:
    position_shot_dash_general[col + '_percentile'] = position_shot_dash_general[col].rank(pct = True, method = 'first')

# plot position FGA vs EFG% with plotly
filt_position_shot_dash_general = position_shot_dash_general[position_shot_dash_general['FGA'] > 1]
fig = px.scatter(filt_position_shot_dash_general, x = 'FGA', y = 'EFG%', 
                 hover_name = 'PLAYER', color_discrete_sequence = px.colors.qualitative.Dark24,
                 height= 800, width = 800)
fig.update_layout(title = 'Field Goal Attempts vs Effective Field Goal % for Position ')
# make plot bigger
fig.update_traces(marker = dict(size = 10))

# add average for x and y
x_avg = position_shot_dash_general_comp['FGA'].mean()
y_avg = position_shot_dash_general_comp['EFG%'].mean()

player_photo =st.session_state.player_photo

# add player photo
with open(player_photo, "rb") as f:
    encoded_image = base64.b64encode(f.read()).decode("ascii")

# add player photo to the plot
fig.add_layout_image(
    dict(
        source='data:image/png;base64,{}'.format(encoded_image),
        yref="y",
        xref = "x",
        y=position_shot_dash_general_comp['EFG%'].values[0],
        x = position_shot_dash_general_comp['FGA'].values[0],
        sizex=6,  # adjust image size as needed
        sizey=6,  # adjust image size as needed
        xanchor="center",
        yanchor="middle",
        opacity=1,
        layer="above")
)

st.plotly_chart(fig, use_container_width = True)



# Add a plotly chart for the distribution of a chosen data point
#data_point = 'EFG%'

st.write('---')
st.subheader('Select a data point to plot the distribution for the position')

# Add violin plot for the distribution of a chosen data point
data_point = st.selectbox('Choose a data point to plot', position_shot_dash_general.columns[1:10])



st.subheader('Distribution of ' + str(data_point) + ' for Position')

# get player data point
player_datapoint = position_shot_dash_general_comp[data_point].values[0]

fig = px.violin(position_shot_dash_general, y = data_point,
                color_discrete_sequence = px.colors.qualitative.Dark24, 
                box = True, points = 'all', hover_data = ['PLAYER', data_point], 
                width=800, height=800)

fig.update_layout(title = 'Distribution of ' + str(data_point))



with open(player_photo, "rb") as f:
    encoded_image = base64.b64encode(f.read()).decode("ascii")

# add player photo to the plot
fig.add_layout_image(
    dict(
        source='data:image/png;base64,{}'.format(encoded_image),
        yref="y",
        xref = "paper",
        y=player_datapoint,
        x = 0.125,
        sizex=3,  # adjust image size as needed
        sizey=3,  # adjust image size as needed
        xanchor="center",
        yanchor="middle",
        opacity=1,
        layer="above")
)

st.plotly_chart(fig, use_container_width = True)







player_specific_shottype = 'https://www.nba.com/stats/player/' + str(player_nba_id)+'/shooting'
st.sidebar.markdown('[Specific Shot Types]('+ player_specific_shottype + ')')

st.sidebar.markdown('[NBA Stats Glossary](https://www.nba.com/stats/help/glossary)')

# 2 columns
col1, col2 = st.columns(2)


# fix column names in player_gbg
old_col_names = ['trad_player', 'trad_team', 'trad_match up', 'trad_game date', 
                 'trad_w/l', 'trad_min', 'trad_pts', 'trad_fgm', 'trad_fga', 'trad_fg%', 
                 'trad_3pm', 'trad_3pa', 'trad_3p%', 'trad_ftm', 'trad_fta', 'trad_ft%', 
                 'trad_oreb', 'trad_dreb', 'trad_reb', 'trad_ast', 'trad_stl', 'trad_blk', 
                 'trad_tov', 'trad_pf', 'trad_+/-', 'trad_season', 'trad_season_type', 
                 'adv_player', 'adv_team', 'adv_match up', 'adv_game date', 'adv_w/l', 
                 'adv_min', 'adv_offrtg', 'adv_defrtg', 'adv_netrtg', 'adv_ast%', 
                 'adv_ast/to', 'adv_ast\xa0ratio', 'adv_oreb%', 'adv_dreb%', 
                 'adv_reb%', 'adv_to\xa0ratio', 'adv_efg%', 'adv_ts%', 'adv_usg%', 
                 'adv_pace', 'adv_pie', 'adv_season', 'adv_season_type']

new_col_names = ['player', 'team', 'match up', 'game date', 'w/l', 'min', 'pts', 'fgm',
                    'fga', 'fg%', '3pm', '3pa', '3p%', 'ftm', 'fta', 'ft%', 'oreb', 'dreb',
                    'reb', 'ast', 'stl', 'blk', 'tov', 'pf', '+/-', 'season', 'season_type',
                    'adv_player', 'adv_team', 'adv_match up', 'adv_game date', 'adv_w/l',
                    'adv_min', 'offrtg', 'defrtg', 'netrtg', 'ast%', 'ast/to', 'ast\xa0ratio',
                    'oreb%', 'dreb%', 'reb%', 'to\xa0ratio', 'efg%', 'ts%', 'usg%', 'pace',
                    'pie', 'adv_season', 'adv_season_type']

player_gbg.columns = new_col_names

# # plot shots taken vs points scored with a line of best fit in plotly
# fig = px.scatter(player_gbg, x = 'fga', y = 'pts', trendline = 'ols', color = 'Home', hover_name = 'Date', color_discrete_sequence = ['darkblue', 'llightblue'])
# fig.update_layout(title = 'Shots Taken vs Points Scored for ' + player)
# fig.update_traces(marker = dict(size = 10))
# # add r squared
# z = np.polyfit(player_gbg['fga'], player_gbg['pts'], 1)
# p = np.poly1d(z)
# r_squared = r2_score(player_gbg['pts'], p(player_gbg['fga']))
# fig.add_annotation(x = player_gbg['fga'].max(), y = player_gbg['pts'].max(), text = 'R Squared: ' + str(round(r_squared, 2)), showarrow = False)
# col1.plotly_chart(fig, use_container_width = True)
# set color options

# try 2 with color options

# # plot shots taken vs points scored with a line of best fit in plotly
# fig = px.scatter(player_gbg, x='fga', y='pts', trendline='ols', hover_name='game date',  color_discrete_sequence = px.colors.qualitative.Dark24)
# fig.update_layout(title='Shots Taken vs Points Scored for ' + player)
# fig.update_traces(marker=dict(size=10))

# # add r squared
# z = np.polyfit(player_gbg['fga'], player_gbg['pts'], 1)
# p = np.poly1d(z)
# r_squared = r2_score(player_gbg['pts'], p(player_gbg['fga']))
# fig.add_annotation(x=player_gbg['fga'].max(), y=player_gbg['pts'].max(),
#                     text='R Squared: ' + str(round(r_squared, 2)), showarrow=False)

# col1.plotly_chart(fig, use_container_width=True, theme = None)


# # plot minutes played vs points scored with a line of best fit in plotly
# fig = px.scatter(player_gbg, x = 'min', y = 'pts', trendline = 'ols',  hover_name = 'game date', color_discrete_sequence = px.colors.qualitative.Dark24)
# fig.update_layout(title = 'Minutes Played vs Points Scored for ' + player)
# fig.update_traces(marker = dict(size = 10))
# # add r squared
# z = np.polyfit(player_gbg['min'], player_gbg['pts'], 1)
# p = np.poly1d(z)
# r_squared = r2_score(player_gbg['pts'], p(player_gbg['min']))
# fig.add_annotation(x = player_gbg['min'].max(), y = player_gbg['pts'].max(), text = 'R Squared: ' + str(round(r_squared, 2)), showarrow = False)
# col2.plotly_chart(fig, use_container_width = True)
