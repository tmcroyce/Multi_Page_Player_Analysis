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
import base64

st.set_page_config(page_title='Player Shooting Tool', page_icon=None, layout="wide", initial_sidebar_state="auto" )


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

#today = pst.strftime('%Y-%m-%d')

# Total
custom_background = """
<style>
[data-testid="stAppViewContainer"] {
background: linear-gradient(to right, #2c3333, #35363C);
}
</style>
"""

# Inject the custom CSS into the Streamlit app
st.markdown(custom_background, unsafe_allow_html=True)


# Define custom CSS for the gradient background
custom_header = """
<style>
[data-testid="stHeader"] {
background: linear-gradient(to right, #2c3333, #35363C);
}
</style>
"""

# Inject the custom CSS into the Streamlit app
st.markdown(custom_header, unsafe_allow_html=True)


custom_metric = """
<style>
[data-testid="metric-container"] {
background: linear-gradient(to right, #35363C, #0e1117);
box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5); /* Add 3D shadow effect */
text-align: center;
max-width: 80%;
}
</style>
"""
st.markdown(custom_metric, unsafe_allow_html=True)


custom_plotly = """
<style>
[class="user-select-none svg-container"] {
background: linear-gradient(to right, #35363C, #0e1117);
border-radius: 30px;  /* Adjust this value to change the rounding of corners */
text-align: center;  /* Center the text inside the metric box */
box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5); /* Add 3D shadow effect */

}
</style>
"""

# Inject the custom CSS into the Streamlit app
st.markdown(custom_plotly, unsafe_allow_html=True)

custom_sidebar = """
<style>
section[data-testid="stSidebar"]{
background-image: linear-gradient(#35363C, #0e1117);
color: white;
}
</style>
"""
st.markdown(custom_sidebar , unsafe_allow_html=True)



files_in_dir = os.listdir('data/player/nba_com_playerdata/tracking')
files_in_dir = [file for file in files_in_dir if file.endswith('.csv')]
# only keep last 14 digits
files_in_dir = [file[-15:] for file in files_in_dir]
# drop last 5 digits
files_in_dir = [file[:-5] for file in files_in_dir]
# sort by most recent date
files_in_dir.sort()
# get the LAST file name
today = files_in_dir[-1]

# Load Data

player_numbers = pd.read_csv('data/player/nba_com_info/players_and_photo_links.csv')
# add capitalized player name
player_numbers['Player'] = player_numbers['player_name'].apply(clean_name)

# Load Sizes
df_sizes = pd.read_csv('data/player/aggregates_of_aggregates/New_Sizes_and_Positions.csv')
# fix names
df_sizes['player'] = df_sizes['player'].apply(clean_name)

# load game by game data
gbg_df = pd.read_csv('data/player/aggregates/Trad&Adv_box_scores_GameView.csv')
# fix names
gbg_df['trad_player'] = gbg_df['trad_player'].apply(clean_name)

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


st.subheader('Ball Handling & Assists')

# set column sizes, middle one small
col1, col2, col3, col4 = st.columns((1, .2, 1, .5))

# make sure position_season is 2022
primary_positions = primary_positions[primary_positions['position_season'] == 2022]
# drop duplicates
primary_positions = primary_positions.drop_duplicates(subset = 'player')

# add position column to gbg_22
gbg_22['position'] = gbg_22['trad_player'].map(primary_positions.set_index('player')['primary_position_bbref'])


# calculate season averages by player from gbg_22, but keep the position column
position_season_averages = gbg_22.groupby(['trad_player', 'position']).mean().reset_index()

# filter by position
position_avg= position_season_averages[position_season_averages['position'] == position]
player_avg = position_season_averages[position_season_averages['trad_player'] == player]

# plot scatterplot of assist ratio to turnover%
fig = px.scatter(position_avg, x = 'adv_ast%', y = 'adv_ast/to', hover_name = 'trad_player', 
                                                opacity = 0.5, size_max = 40, width = 800, height = 1000)
fig.update_layout(title = 'Assist Ratio vs Turnover% by Position')
fig.update_layout(showlegend = False)

# update x and y axis titles
x_title = 'Assist Percent (of Team total)'
y_title = 'Assist / Turnover Ratio'
fig.update_layout(xaxis_title = x_title, yaxis_title = y_title)

# # add player scatter point to plot
# fig.add_trace(go.Scatter(x = player_avg['adv_ast%'], y = player_avg['adv_ast/to'], mode = 'markers',
#                         marker = dict(size = 20, color = 'red'), name = player))

# Add player photo to scatter
player_photo =st.session_state.player_photo


player_assist_to_ratio = player_avg['adv_ast/to'].iloc[0]
player_adv_ast_percent = player_avg['adv_ast%'].iloc[0]

# add player photo
with open(player_photo, "rb") as f:
    encoded_image = base64.b64encode(f.read()).decode("ascii")

# add player photo to the plot
fig.add_layout_image(
    dict(
        source='data:image/png;base64,{}'.format(encoded_image),
        yref="y",
        xref = "x",
        y = player_assist_to_ratio,
        x = player_adv_ast_percent,
        sizex=10,  # adjust image size as needed
        sizey=10,  # adjust image size as needed
        xanchor="center",
        yanchor="middle",
        opacity=1,
        layer="above")
)

fig.update_layout(# transparent background and paper
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
col1.plotly_chart(fig, use_container_width = True)

# calculate percentiles for player in position

# check to see if player is in position_avg
if player not in position_avg['trad_player'].values:
    # add player_avg
    position_avg = position_avg.append(player_avg)

col3.write('**Player Metrics & Percentiles**')

# make sure adv_ast/to is float
position_avg['adv_ast/to'] = position_avg['adv_ast/to'].astype(float)


position_avg['adv_ast%_percentile'] = position_avg['adv_ast%'].rank(pct = True)
position_avg['adv_ast/to_percentile'] = position_avg['adv_ast/to'].rank(pct = True)

player_ast_percent_percentile = position_avg[position_avg['trad_player'] == player]['adv_ast%_percentile'].values[0]
player_ast_to_percentile = position_avg[position_avg['trad_player'] == player]['adv_ast/to_percentile'].values[0]
player_filtered_avg = position_avg[position_avg['trad_player'] == player]

# write assist percent
col3.write(' ')
col3.write(' ')
col3.write(' ')
col3.write('**Player Assist Percent:** ' + str(round(player_filtered_avg['adv_ast%'].values[0], 2)) + '%')
col3.write('Assist percent is the percentage of team assists a player has when he is on the floor.')
col3.write('**Assist Percentile:** ' + str(round(player_ast_percent_percentile*100, 2)) + '%')
col3.write(' ')
col3.write(' ')
col3.write(' ')
col3.write(' ')
col3.write(' ')
col3.write(' ')
col3.write(' ')


# add donut chart of position_ast_percent_percentile
fig = go.Figure(data = [go.Pie(labels = ['Percentile', ''], values = [player_ast_percent_percentile, 1-player_ast_percent_percentile], hole = 0.75)])
# make smaller
fig.update_layout(width = 250, height = 250)
# no legend
fig.update_layout(showlegend = False)
fig.update_layout(# transparent background and paper
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
# do not show values
fig.update_traces(textinfo = 'none')
# add outline to donut
fig.update_traces(marker_line_width = 1, marker_line_color = 'black')
# color red if percentile is less than 0.5, green if greater than 0.5
if player_ast_percent_percentile < 0.5:
    fig.update_traces(marker = dict(colors = ['red', '#ffffff']))
else:
    fig.update_traces(marker = dict(colors = ['green', '#ffffff']))

col4.plotly_chart(fig, use_container_width = True)


# write assist to
col3.write(' ')
col3.write('**Player Assist / Turnover Ratio:** ' + str(round(player_filtered_avg['adv_ast/to'].values[0], 2)))
col3.write('Assist / Turnover Ratio is the ratio of assists to turnovers a player has when he is on the floor.')
col3.write('**Assist / Turnover Ratio Percentile:** ' + str(round(player_ast_to_percentile*100, 2)) + '%')
col3.write(' ')
col3.write(' ')
col3.write(' ')
col3.write(' ')
col3.write(' ')
col3.write(' ')
col3.write(' ')
col3.write(' ')
col3.write(' ')

# add donut chart of position_ast_to_percentile
fig = go.Figure(data = [go.Pie(labels = ['Percentile', ''], values = [player_ast_to_percentile, 1-player_ast_to_percentile], hole = 0.75)])
# make smaller
fig.update_layout(width = 250, height = 250)
# no legend
fig.update_layout(showlegend = False)
# do not show values
fig.update_traces(textinfo = 'none')
# add outline to donut
fig.update_traces(marker_line_width = 1, marker_line_color = 'black')
fig.update_layout(# transparent background and paper
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
# color red if percentile is less than 0.5, green if greater than 0.5
if player_ast_to_percentile < 0.5:
    fig.update_traces(marker = dict(colors = ['red', '#ffffff']))
else:
    fig.update_traces(marker = dict(colors = ['green', '#ffffff']))

col4.plotly_chart(fig, use_container_width = True)





# # add metrics
# player_ast_percent = player_filtered_avg['adv_ast%'].values[0]
# col3.metric('Assist Percent (Percent of Team Assists when on Floor)', value = player_ast_percent.round(1), delta = (str(player_ast_percent_percentile.round(2)*100) + ' percentile (Higher is better)'))
# player_ast_to = player_filtered_avg['adv_ast/to'].values[0]
# col3.metric('Assist / Turnover Ratio', value = player_ast_to.round(1), delta = (str(player_ast_to_percentile.round(2) *100) + ' percentile (Higher is better)'))
# find the column with 'to' and 'ratio' in it
to_ratio_col = [col for col in player_filtered_avg.columns if 'to' in col and 'ratio' in col][0]

# add percentile to position_avg
position_avg['adv_to ratio percentile'] = position_avg[to_ratio_col].rank(pct = True)
player_filtered_avg = position_avg[position_avg['trad_player'] == player]

advanced_to_ratio = player_filtered_avg[to_ratio_col].values[0]

# write adv turnover ratio
col3.write('**Player Turnover Ratio:** ' + str(round(advanced_to_ratio, 2)))
col3.write('Turnover Ratio is the ratio of turnovers to possessions a player has when he is on the floor. A lower percentile is better.')
col3.write('**Turnover Ratio Percentile:** ' + str(round(player_filtered_avg['adv_to ratio percentile'].values[0]*100, 2)) + '%')
col3.write(' ')
col3.write(' ')
col3.write(' ')
col3.write(' ')
col3.write(' ')
col3.write(' ')
col3.write(' ')


# add donut chart of adv_to_ratio_percentile
fig = go.Figure(data = [go.Pie(labels = ['Percentile', ''], values = [player_filtered_avg['adv_to ratio percentile'].values[0], 1-player_filtered_avg['adv_to ratio percentile'].values[0]], hole = 0.75)])
# make smaller
fig.update_layout(width = 300, height = 250)
# no legend
fig.update_layout(showlegend = False)
fig.update_layout(# transparent background and paper
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
# do not show values
fig.update_traces(textinfo = 'none')
# add outline to donut
fig.update_traces(marker_line_width = 1, marker_line_color = 'black')
# color red if percentile is less than 0.5, green if greater than 0.5

if player_filtered_avg['adv_to ratio percentile'].values[0] < 0.5:
    fig.update_traces(marker = dict(colors = ['red', '#ffffff']))
else:
    fig.update_traces(marker = dict(colors = ['green', '#ffffff']))

col4.plotly_chart(fig, use_container_width = True)



#col3.metric('Turnover Ratio (Average Player Turnovers per 100 Possessions)', value = advanced_to_ratio.round(1), delta = (str(player_filtered_avg['adv_to ratio percentile'].values[0].round(2) *100) + ' percentile (Lower is better)'))

position_avg['ast_percentile'] = position_avg['trad_ast'].rank(pct = True)
player_filtered_avg = position_avg[position_avg['trad_player'] == player]
player_ast = player_filtered_avg['trad_ast'].values[0]

# write assist per game
col3.write('**Player Assists per Game:** ' + str(round(player_ast, 2)))
col3.write('Assists per Game is the average number of assists a player has per game.')
col3.write('**Assist per Game Percentile:** ' + str(round(player_filtered_avg['ast_percentile'].values[0]*100, 2)) + '%')



# add donut chart of ast_percentile
fig = go.Figure(data = [go.Pie(labels = ['Percentile', ''], values = [player_filtered_avg['ast_percentile'].values[0], 1-player_filtered_avg['ast_percentile'].values[0]], hole = 0.75)])
# make smaller
fig.update_layout(width = 300, height = 250)
# no legend
fig.update_layout(showlegend = False)
# do not show values
fig.update_traces(textinfo = 'none')
fig.update_layout(# transparent background and paper
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
# add outline to donut
fig.update_traces(marker_line_width = 1, marker_line_color = 'black')
# color red if percentile is less than 0.5, green if greater than 0.5
if player_filtered_avg['ast_percentile'].values[0] < 0.5:
    fig.update_traces(marker = dict(colors = ['red', '#ffffff']))
else:
    fig.update_traces(marker = dict(colors = ['green', '#ffffff']))

col4.plotly_chart(fig, use_container_width = True)