import streamlit as st
import pandas as pd
import numpy as np
import datetime as datetime
import plotly as py
import plotly.graph_objs as go
import plotly.express as px
import sklearn as sk
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
import plotly.figure_factory as ff
import unidecode
import re
import os

st.set_page_config(page_title='Player Game Data Tool', page_icon=None, layout="wide", initial_sidebar_state="auto" )

def add_css():

    #### ADD CSS STYLING ####
    custom_background = """
    <style>
    [data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #2c3333, #35363C);
    }
    </style>
    """

    # Inject the custom CSS into the Streamlit app
    st.sidebar.markdown(custom_background, unsafe_allow_html=True)



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

add_css()

st.markdown(f"""
    <h1 style="
        font-family: Arial, sans-serif;
        font-size: 48px;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        padding: 10px;
        background: linear-gradient(to right, {st.session_state['title_color_1']}, {st.session_state['title_color_2']});
        box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5); /* Add 3D shadow effect */
        border-radius: 20px;
    ">Recent Game Data</h1>
""", unsafe_allow_html=True)

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

###### LOAD DATA -- SAME EVERY PAGE #############################################################################

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

# check last date
gbg_df['Date'] = pd.to_datetime(gbg_df['trad_game date'])
# change datetime to date
gbg_df['Date'] = gbg_df['Date'].dt.date
# sort by date
gbg_df = gbg_df.sort_values(by = 'Date', ascending = False)
# get last date
last_date = gbg_df['Date'].iloc[0]


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


##### LOAD DATA END #############################################################################


st.write('')

player_gbg_22 = player_gbg[player_gbg['adv_season'] == 2022]


# get season averages
player_ppg = round(player_gbg_22['trad_pts'].sum() / len(player_gbg_22),1)
player_rpg = round(player_gbg_22['trad_reb'].sum() / len(player_gbg_22),1)
player_apg = round(player_gbg_22['trad_ast'].sum() / len(player_gbg_22),1)
player_tovpg = round(player_gbg_22['trad_tov'].sum() / len(player_gbg_22),1)
player_3p_pct = round(player_gbg_22['trad_3pm'].sum() / player_gbg_22['trad_3pa'].sum() *100, 1)
player_fg_pct = round(player_gbg_22['trad_fgm'].sum() / player_gbg_22['trad_fga'].sum() *100, 1)


# add home or away column
player_gbg['Home'] = np.where(player_gbg['trad_match up'].str.contains('vs'), 1, 0)
# add points per minute column
player_gbg['ppm'] = player_gbg['trad_pts'] / player_gbg['trad_min']
# fix columns names, drop 'trad'
player_gbg.columns = player_gbg.columns.str.replace('trad_', '')
# get numeric columns
numeric_cols= player_gbg.select_dtypes(include = ['float64', 'int64']).columns
to_drop = ['adv_match up', 'adv_game date', 'adv_w/l', 'adv_player', 'adv_team', 'adv_season_type']
player_gbg = player_gbg.drop(columns = to_drop)
# set index to player
player_gbg = player_gbg.set_index('player')

# fix game date, sort by game date
player_gbg['game_date'] = pd.to_datetime(player_gbg['game date']).dt.date
player_gbg = player_gbg.sort_values(by = 'game_date', ascending = False)



# calculate averages for player
player_gbg_avg = player_gbg.groupby('player').mean().reset_index()
 
 # show efg% and ts%, 3p%, fg%, adv_pace
player_gbg_avg = player_gbg_avg[['player', 'adv_offrtg', 'adv_defrtg', 'adv_efg%', 'adv_ts%',  'adv_pace', 'ppm']]
ncols = ['adv_offrtg', 'adv_defrtg', 'adv_efg%', 'adv_ts%',  'adv_pace', 'ppm']
# get rid of index
player_gbg_avg = player_gbg_avg.set_index('player')
# rename columns
player_gbg_avg.columns = ['Offensive Rating', 'Defensive Rating', 'Effective FG%', 'True Shooting%', 'Pace', 'Points Per Minute']



# make sure position_season is 2022
primary_positions = primary_positions[primary_positions['position_season'] == 2022]
# drop duplicates
primary_positions = primary_positions.drop_duplicates(subset = 'player')

# add position column to gbg_22
gbg_22['position'] = gbg_22['trad_player'].map(primary_positions.set_index('player')['primary_position_bbref'])

# compare player to position

# get averages for player position
position_gbg = gbg_22[gbg_22['position'] == position]
position_gbg_mean = position_gbg.groupby('position').mean().reset_index()

# make adv_season an int
player_gbg['adv_season'] = player_gbg['adv_season'].astype(int)



# Add Last 10 Games #####################


st.subheader('Last N Games')

# select number of games to show
n = st.slider('Number of Games', 1, 82, 11)

# get last 10 games by using .head(10)

last_10 = player_gbg.head(n)

# drop Unnamed columns
unnamed = [col for col in last_10.columns if 'Unnamed' in col]
last_10.drop(columns = unnamed, inplace = True)

# get numeric cols
num_cols = last_10.columns[4:]
# make sure all num_cols are numeric
last_10[num_cols] = last_10[num_cols].apply(pd.to_numeric, errors = 'coerce')
# drop season_type column
last_10.drop(columns = ['season_type', 'adv_min', 'adv_season'], inplace = True)
num_cols = last_10.columns[4:]

# add metrics, compare to season average
last_10_ppg = last_10['pts'].mean()
last_10_ppm = last_10['ppm'].mean()
last_10_3p = last_10['3pm'].sum() / last_10['3pa'].sum() *100
last_10_ast = last_10['ast'].mean()
last_10_reb = last_10['reb'].mean()
last_10_stl = last_10['stl'].mean()
last_10_reb = last_10['reb'].mean()
last_10_efg = last_10['adv_efg%'].mean()
last_10_ts = last_10['adv_ts%'].mean()
last_10_usg = last_10['adv_usg%'].mean()

player_gbg_22 = player_gbg[player_gbg['adv_season'] == 2022]

season_ppg = player_gbg_22['pts'].mean()
season_ppm = player_gbg_22['ppm'].mean()
season_3p = player_gbg_22['3pm'].sum() / player_gbg_22['3pa'].sum() *100
season_ast = player_gbg_22['ast'].mean()
season_reb = player_gbg_22['reb'].mean()
season_stl = player_gbg_22['stl'].mean()
season_reb = player_gbg_22['reb'].mean()
season_efg = player_gbg_22['adv_efg%'].mean()
season_ts = player_gbg_22['adv_ts%'].mean()
season_usg = player_gbg_22['adv_usg%'].mean()


# First row with 5 columns
col1, col2, col3, col4 = st.columns([.25, .25, .25, .25])
col1.metric(label='Points Per Game', value=round(last_10_ppg, 1), delta=round(last_10_ppg - season_ppg, 1))
col2.metric(label='Points Per Minute', value=round(last_10_ppm, 1), delta=round(last_10_ppm - season_ppm, 1))
col3.metric(label='3P%', value=round(last_10_3p, 1), delta=round(last_10_3p - season_3p, 1))
col4.metric(label='AST', value=round(last_10_ast, 1), delta=round(last_10_ast - season_ast, 1))

# Second row with 4 columns
col5, col6, col7, col8 = st.columns([.25, .25, .25, .25])
col5.metric(label='REB', value=round(last_10_reb, 1), delta=round(last_10_reb - season_reb, 1))
col6.metric(label='STL', value=round(last_10_stl, 1), delta=round(last_10_stl - season_stl, 1))
col7.metric(label='eFG%', value=round(last_10_efg, 1), delta=round(last_10_efg - season_efg, 1))
col8.metric(label='USG%', value=round(last_10_usg, 1), delta=round(last_10_usg - season_usg, 1))

# Display last 10 games
# st.dataframe(last_10.style.format('{:.1f}', subset = num_cols))
st.write('')
st.write('Over the past '+ str(n) +' games, ' + player + ' has scored at a pace of ' + str(round(last_10['ppm'].mean(), 1)) + ' points per minute, ' + str(round(last_10['pts'].mean(), 1)) + ' points per game, while shooting ' + str(round(last_10['3p%'].mean(), 1)) + '% from three.')
st.write('He is averaging ' + str(round(last_10['ast'].mean(), 1)) + ' assists per game, ' + str(round(last_10['reb'].mean(), 1)) + ' rebounds per game, and ' + str(round(last_10['stl'].mean(), 1)) + ' steals per game.')

# Visualize PPM, Points, and 3P% over last 10 games
colz  = st.columns(3)
col1 = colz[0]
col2 = colz[1]
col3 = colz[2]

def get_marker_color(value, cmap, vmin, vmax):
    norm = (value - vmin) / (vmax - vmin)
    idx = int(norm * (len(cmap) - 1))
    return cmap[idx]

cmap = px.colors.diverging.BrBG
vmin = last_10['ppm'].min()
vmax = last_10['ppm'].max()
marker_colors = [get_marker_color(value, cmap, vmin, vmax) for value in last_10['ppm']]

fig = px.line(last_10, x='game date', y='ppm', title='PPM over Last 10 Games', 
              markers=True)
fig.update_traces(line=dict(color='grey'), marker=dict(color=marker_colors), selector=dict(type='scatter', mode='markers'))
fig.update_layout(xaxis_title='Date', yaxis_title='PPM')
fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
                  paper_bgcolor='rgba(0, 0, 0, 0)')
# # add player average
fig.add_hline(y = last_10['ppm'].mean(), line_dash = 'dash', line_color = 'red')
col1.plotly_chart(fig, use_container_width = True)


############ POINTS ###########
cmap = px.colors.diverging.BrBG
vmin = last_10['pts'].min()
vmax = last_10['pts'].max()
marker_colors = [get_marker_color(value, cmap, vmin, vmax) for value in last_10['pts']]

fig = px.line(last_10, x='game date', y='pts', title='Points over Last 10 Games', 
              markers=True)
fig.update_traces(line=dict(color='grey'), marker=dict(color=marker_colors), selector=dict(type='scatter', mode='markers'))
fig.update_layout(xaxis_title='Date', yaxis_title='Points')
fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
                  paper_bgcolor='rgba(0, 0, 0, 0)')
# add player average
fig.add_hline(y = last_10['pts'].mean(), line_dash = 'dash', line_color = 'red')
col2.plotly_chart(fig, use_container_width = True)


line_color_3p = px.colors.diverging.BrBG[3]  # Select a color from the BrBG color scale for 3P% line
marker_color_3p = 'rgba(34, 97, 153, 0.6)'  # Common marker color for 3P% line

line_color_efg = px.colors.diverging.BrBG[-4]  # Select a color from the BrBG color scale for EFG% line
marker_color_efg = 'rgba(153, 34, 34, 0.6)'  # Common marker color for EFG% line

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=last_10['game date'],
    y=last_10['3p%'],
    mode='lines+markers',
    name='3P%',
    line=dict(color=line_color_3p),
    marker=dict(color=marker_color_3p)
))

fig.add_trace(go.Scatter(
    x=last_10['game date'],
    y=last_10['adv_efg%'],
    mode='lines+markers',
    name='EFG%',
    line=dict(color=line_color_efg),
    marker=dict(color=marker_color_efg)
))

fig.update_layout(
    title='3P% and EFG% over Last 10 Games',
    xaxis_title='Date',
    yaxis_title='Percentage',
    plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
    paper_bgcolor='rgba(0, 0, 0, 0)'  # Transparent paper background
)


# add player average
fig.add_hline(y = last_10['3p%'].mean(), line_dash = 'dash', line_color = 'red')
col3.plotly_chart(fig, use_container_width = True)


# plot usage distribution over last 10
fig = px.violin(last_10, x = 'adv_usg%', title = 'Usage Distribution over Last 10 Games', box = True, points = 'all', hover_data = last_10.columns)
fig.update_layout(xaxis_title = 'Usage %', yaxis_title = 'Frequency')
fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
                        paper_bgcolor='rgba(0, 0, 0, 0)')
col1.plotly_chart(fig, use_container_width = True)


# plot offensive rating distribution over last 10
fig = px.violin(last_10, x = 'adv_offrtg', title = 'Offensive Rating Distribution over Last 10 Games', box = True, points = 'all', hover_data = last_10.columns)
fig.update_layout(xaxis_title = 'Offensive Rating', yaxis_title = 'Frequency')
fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
                        paper_bgcolor='rgba(0, 0, 0, 0)')
col2.plotly_chart(fig, use_container_width = True)


# plot defensive rating distribution over last 10
fig = px.violin(last_10, x = 'adv_defrtg', title = 'Defensive Rating Distribution over Last 10 Games', box = True, points = 'all', hover_data = last_10.columns)
fig.update_layout(xaxis_title = 'Defensive Rating', yaxis_title = 'Frequency')
fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
                        paper_bgcolor='rgba(0, 0, 0, 0)')
col3.plotly_chart(fig, use_container_width = True)











