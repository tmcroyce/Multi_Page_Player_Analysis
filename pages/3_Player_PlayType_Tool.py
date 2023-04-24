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
# Total




# get current time in pst
pst = datetime.timezone(datetime.timedelta(hours=-8))
# to datetime
pst = datetime.datetime.now(pst)

#today = pst.strftime('%Y-%m-%d')

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

# Load Data



player_numbers = pd.read_csv('data/player/nba_com_info/players_and_photo_links.csv')

# fix player_numbers names
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

# load game by game data
gbg_df = pd.read_csv('data/player/aggregates/Trad&Adv_box_scores_GameView.csv')
# fix names
gbg_df['trad_player'] = gbg_df['trad_player'].apply(clean_name)

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

st.markdown(f"""
    <h1 style="
        font-family: Arial, sans-serif;
        font-size: 48px;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        padding: 20px;
        background: linear-gradient(to right, {st.session_state['title_color_1']}, {st.session_state['title_color_2']});;
        box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5); /* Add 3D shadow effect */
        border-radius: 30px;
    ">Player Playtype Data</h1>
""", unsafe_allow_html=True)


playtype_folder = 'data/player/nba_com_playerdata/playtypes/'
pt_fold = os.listdir(playtype_folder)
# check if today's date is in file names
if 'pt_cut_' + str(today) + '_.csv' in pt_fold:
    # read in today's data
    pt_cut = pd.read_csv(playtype_folder + 'pt_cut_' + str(today) + '_.csv')
    pt_cut['PLAYER'] = pt_cut['PLAYER'].apply(clean_name)

    pt_hand_off = pd.read_csv(playtype_folder + 'pt_hand_off_' + str(today) + '_.csv')
    pt_hand_off['PLAYER'] = pt_hand_off['PLAYER'].apply(clean_name)

    pt_isolation = pd.read_csv(playtype_folder + 'pt_isolation_' + str(today) + '_.csv')
    pt_isolation['PLAYER'] = pt_isolation['PLAYER'].apply(clean_name)

    pt_off_screen = pd.read_csv(playtype_folder + 'pt_off_screen_' + str(today) + '_.csv')
    pt_off_screen['PLAYER'] = pt_off_screen['PLAYER'].apply(clean_name)

    pt_post_up = pd.read_csv(playtype_folder + 'pt_post_up_' + str(today) + '_.csv')
    pt_post_up['PLAYER'] = pt_post_up['PLAYER'].apply(clean_name)

    pt_pr_ball_handler = pd.read_csv(playtype_folder + 'pt_pr_ball_handler_' + str(today) + '_.csv')
    pt_pr_ball_handler['PLAYER'] = pt_pr_ball_handler['PLAYER'].apply(clean_name)

    pt_pr_roll_man = pd.read_csv(playtype_folder + 'pt_pr_roll_man_' + str(today) + '_.csv')
    pt_pr_roll_man['PLAYER'] = pt_pr_roll_man['PLAYER'].apply(clean_name)

    pt_spot_up = pd.read_csv(playtype_folder + 'pt_spot_up_' + str(today) + '_.csv')
    pt_spot_up['PLAYER'] = pt_spot_up['PLAYER'].apply(clean_name)

    pt_transition = pd.read_csv(playtype_folder + 'pt_transition_' + str(today) + '_.csv')
    pt_transition['PLAYER'] = pt_transition['PLAYER'].apply(clean_name)

    pt_putbacks = pd.read_csv(playtype_folder + 'pt_putbacks_' + str(today) + '_.csv')
    pt_putbacks['PLAYER'] = pt_putbacks['PLAYER'].apply(clean_name)

else:
    st.write('Todays Data Needs to be Collected')



# Load available playtype data

player_iso = pt_isolation[pt_isolation['PLAYER'] == player]
if player_iso.empty:
    st.write('No Isolation Data for ' + player)
else:
    player_iso.index = ['Isolation Offense']

player_cut = pt_cut[pt_cut['PLAYER'] == player]
if player_cut.empty:
    st.write('No Cut Data for ' + player)
else:
    player_cut.index = ['Cut Offense']

player_hand_off = pt_hand_off[pt_hand_off['PLAYER'] == player]
if player_hand_off.empty:
    st.write('No Hand Off Data for ' + player)
else:
    player_hand_off.index = ['Hand Off Offense']

player_off_screen = pt_off_screen[pt_off_screen['PLAYER'] == player]
if player_off_screen.empty:
    st.write('No Off Screen Data for ' + player)
else:
    player_off_screen.index = ['Off Screen Offense']

player_post_up = pt_post_up[pt_post_up['PLAYER'] == player]
if player_post_up.empty:
    st.write('No Post Up Data for ' + player)
else:
    player_post_up.index = ['Post Up Offense']

player_pr_ball_handler = pt_pr_ball_handler[pt_pr_ball_handler['PLAYER'] == player]
if player_pr_ball_handler.empty:
    st.write('No Pick and Roll Ball Handler Data for ' + player)
else:
    player_pr_ball_handler.index = ['Pick and Roll Ball Handler Offense']

player_spot_up = pt_spot_up[pt_spot_up['PLAYER'] == player]
if player_spot_up.empty:
    st.write('No Spot Up Data for ' + player)
else:
    player_spot_up.index = ['Spot Up Offense']

player_transition = pt_transition[pt_transition['PLAYER'] == player]
if player_transition.empty:
    st.write('No Transition Data for ' + player)
else:
    player_transition.index = ['Transition Offense']

player_putbacks = pt_putbacks[pt_putbacks['PLAYER'] == player]
if player_putbacks.empty:
    st.write('No Putback Data for ' + player)
else:
    player_putbacks.index = ['Putback Offense']

# combine all the dataframes that have data
playtypes = pd.concat([player_iso, player_cut, player_hand_off, player_off_screen, player_post_up, player_pr_ball_handler, player_spot_up, player_transition, player_putbacks], axis = 0)

unnamed = [col for col in playtypes.columns if 'Unnamed' in col]
playtypes.drop(columns = unnamed, inplace = True)

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

# give option to show table with data
if st.checkbox('Show Playtype Data Table'):
    st.table(playtypes.style.format('{:.2f}', subset= playtypes.columns[3:]).applymap(color_code_percentile2, subset = 'Percentile'))

# identify num cols (last 14 cols)
num_cols = player_iso.columns[-14:]

# add PPP normalized column
playtypes['PPP_norm'] = playtypes['PPP'] / playtypes['PPP'].max()

# plotly scatterplot of FREQ% vs Percentile, sized by PPP (min size 10, max size 50), colored by playtype
fig = px.scatter(playtypes, x = 'Percentile', y = 'Freq%', size = 'PPP_norm', color = playtypes.index, size_max = 50)
                 

fig.update_layout(title = 'Frequency of Playtype vs Percentile for ' + player, height = 600)
fig.update_layout(xaxis_title = 'Player NBA Percentile (How Good They Are At Specified Play)', yaxis_title = 'Frequency of Playtype (How Often They Run Specified Play)',
                  # change background and paper color to transparent
                    paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(0,0,0,0)',
                    # make x axis and y axis labels larger
                    xaxis = dict(title_font = dict(size = 20)),
                    yaxis = dict(title_font = dict(size = 20)))


# add annotations
for i in range(len(playtypes)):
    fig.add_annotation(x = playtypes['Percentile'][i], y = playtypes['Freq%'][i], text = playtypes.index[i])

st.plotly_chart(fig, use_container_width = True)


# add playtype breakdwon donut chart with Freq%
fig = go.Figure(data = [go.Pie(labels = playtypes.index, values = playtypes['Freq%'], hole = 0.6)])
fig.update_layout(title = 'Playtype Breakdown for ' + player, height = 600,
                                    # change background and paper color to transparent
                    paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(0,0,0,0)')
# add annotations
for i in range(len(playtypes)):
    fig.add_annotation(x = playtypes['Percentile'][i], y = playtypes['Freq%'][i], text = playtypes.index[i])
    
st.plotly_chart(fig, use_container_width = True)




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

custom_table = """
<style>
[data-testid="stTable"]{
background: linear-gradient(to right, #35363C, #0e1117);
border-radius: 30px;  /* Adjust this value to change the rounding of corners */
text-align: center;  /* Center the text inside the metric box */
box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5); /* Add 3D shadow effect */
color: white;
}
</style>
"""
st.markdown(custom_table , unsafe_allow_html=True)