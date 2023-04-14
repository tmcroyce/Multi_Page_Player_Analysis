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
import plotly.graph_objs as go
import plotly.express as px
import base64
import unidecode
import re

st.set_page_config(page_title='Player Analyzer Tool', page_icon=None, layout="wide", initial_sidebar_state="auto" )

# get current time in pst
pst = datetime.timezone(datetime.timedelta(hours=-8))
# to datetime
pst = datetime.datetime.now(pst)

st.write(f"Current working directory: {os.getcwd()}")

#today = pst.strftime('%Y-%m-%d')

files_in_dir = os.listdir('app\\multi_page_player_analysis\\data\\player\\nba_com_playerdata\\tracking')
files_in_dir = [file for file in files_in_dir if file.endswith('.csv')]
# only keep last 14 digits
files_in_dir = [file[-15:] for file in files_in_dir]
# drop last 5 digits
files_in_dir = [file[:-5] for file in files_in_dir]
files_in_dir
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

def show_sidebar():
     # Write session states out on sidebar
    st.sidebar.write('Session States')
    st.sidebar.write('Player: ' + st.session_state['player'])
    st.sidebar.write('Player Number: ' + str(st.session_state['player_number']))
    st.sidebar.write('Team: ' + st.session_state['team'])
    st.sidebar.write('Team Number: ' + str(st.session_state['team_num']))
    st.sidebar.write('Position: ' + st.session_state['position'])
    st.sidebar.write('Position Index: ' + str(st.session_state['position_index']))


# Load Data

player_numbers = pd.read_csv('data/player/nba_com_info/players_and_photo_links.csv')
# add capitalized player name
player_numbers['Player'] = player_numbers['player_name'].apply(clean_name)

# Load Sizes
df_sizes = pd.read_csv('data/player/aggregates_of_aggregates/New_Sizes_and_Positions.csv')

# load game by game data
gbg_df = pd.read_csv('data/player/aggregates/Trad&Adv_box_scores_GameView.csv')

# check last date
gbg_df['Date'] = pd.to_datetime(gbg_df['trad_game date'])
# change datetime to date
gbg_df['Date'] = gbg_df['Date'].dt.date
# sort by date
gbg_df = gbg_df.sort_values(by = 'Date', ascending = False)
# get last date
last_date = gbg_df['Date'].iloc[0]

st.title('NBA Player Size Comparison Tool')

st.write('The data for positions is pulled from BasketballReference.com and is the actual position they play on the floor, as opposed to the NBA listed position.')
st.write('---')

st.subheader('Player Size Data')
st.write('')

# load game by game data
gbg_df = pd.read_csv('data/player/aggregates/Trad&Adv_box_scores_GameView.csv')

# fix names in loaded data
gbg_df['trad_player'] = gbg_df['trad_player'].apply(clean_name)
df_sizes['player'] = df_sizes['player'].apply(clean_name)

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

# get all heights for position
all_heights = df_sizes[df_sizes['primary_position_bbref'] == position]['height_final']
all_wingspans = df_sizes[df_sizes['primary_position_bbref'] == position]['wingspan_final']

# calculate wingspan over height in df_sizes
df_sizes['wingspan__height_ratio'] = df_sizes['wingspan_final'] / df_sizes['height_final']
all_wingspan_height_ratios = df_sizes[df_sizes['primary_position_bbref'] == position]['wingspan__height_ratio']

# get percentile of player height
player_height_percentile = norm.cdf(player_height, all_heights.mean(), all_heights.std()) * 100
player_wingspan_percentile = norm.cdf(player_wingspan, all_wingspans.mean(), all_wingspans.std()) * 100
player_wingspan_height_ratio_percentile = norm.cdf(player_wingspan / player_height, all_wingspan_height_ratios.mean(), all_wingspan_height_ratios.std()) * 100



# 3 columns
col1, col2, col3, col4, col5 = st.columns([.3, .05, .3, .05, .3])
# display player size data


def number_post(num):
    # get the last digit of the number
    last_digit = str(num)[-1]
    # if the last digit is 1, return st
    if last_digit == '1':
        return 'st'
    # if the last digit is 2, return nd
    elif last_digit == '2':
        return 'nd'
    # if the last digit is 3, return rd
    elif last_digit == '3':
        return 'rd'
    # if the last digit is 4-9, return th
    else:
        return 'th'


col1.metric('Player Height: ', str(player_height) + ' inches',  delta_color='off')
col1.write('---')
col1.write(player + ' is in the **' + str(int(player_height_percentile)) + number_post(int(player_height_percentile))+'** percentile for height at the ' + position + ' position.')

def color_def():
    if player_height_percentile < 50:
        return 'red'
    else:
        return 'green'


def plot_height_percentile():
    # plot small bar chart for height percentile with plotly. Color is red if below median, green if above median
    fig = go.Figure(go.Bar(x = [player_height_percentile], y = ['Height Percentile'], orientation = 'h', marker_color = color_def()))
    fig.update_layout(title = position + ' Height Percentile', width = 400, height = 200)
    # show the whole bar chart
    # get rid of y axis
    fig.update_yaxes(showticklabels = False)
    fig.update_xaxes(range = [0, 100])
    # add x-axis ticks
    fig.update_xaxes(tickmode = 'array', tickvals = [0, 20, 40, 60, 80, 100])
    # add gridlines
    fig.update_xaxes(showgrid = True, gridwidth = 1, gridcolor = 'grey')
    # add outline to bar
    fig.update_traces(marker_line_width = 1, marker_line_color = 'black')
    return col1.plotly_chart(fig, use_container_width = True)

plot_height_percentile()

def plot_wingspan_percentile():
    col3.metric('Player Wingspan', str(player_wingspan) + ' inches')
    col3.write('---')
    col3.write(player + ' is in the **' + str(int(player_wingspan_percentile)) + number_post(int(player_wingspan_percentile))+'** percentile for wingspan at the ' + position + ' position.')
    
    def color_def():
        if player_wingspan_percentile < 50:
            return 'red'
        else:
            return 'green'

    # plot small bar chart for wingspan percentile with plotly. Color is red if below median, green if above median
    fig = go.Figure(go.Bar(x = [player_wingspan_percentile], y = ['Wingspan Percentile'], orientation = 'h', marker_color = color_def()))
    fig.update_layout(title = position + ' Wingspan Percentile ', width = 400, height = 200)
    # show the whole bar chart
    # add x-axis ticks
    fig.update_xaxes(range = [0, 100])
    fig.update_xaxes(tickmode = 'array', tickvals = [0, 20, 40, 60, 80, 100])
    # add gridlines
    fig.update_xaxes(showgrid = True, gridwidth = 1, gridcolor = 'grey')
    # get rid of y axis
    fig.update_yaxes(showticklabels = False)
    fig.update_traces(marker_line_width = 1, marker_line_color = 'black')

    return col3.plotly_chart(fig, use_container_width = True)

plot_wingspan_percentile()


def plot_wing_height_rate():

    col5.metric('Player Wingspan / Height Ratio', str(round(player_wingspan / player_height,2)))
    col5.write('---')
    col5.write(player + ' is in the **' + str(int(player_wingspan_height_ratio_percentile)) + number_post(int(player_wingspan_height_ratio_percentile))+'** percentile for wingspan / height ratio at the ' + position + ' position.')

    def color_def():
        if player_wingspan_height_ratio_percentile < 50:
            return 'red'
        else:
            return 'green'

    # plot small bar chart for wingspan / height ratio percentile with plotly. Color is red if below median, green if above median
    fig = go.Figure(go.Bar(x = [player_wingspan_height_ratio_percentile], y = ['Wingspan / Height Ratio Percentile'], orientation = 'h', marker_color = color_def()))
    fig.update_layout(title = position + ' Wingspan / Height Ratio Percentile', width = 400, height = 200)
    # show the whole bar chart
    fig.update_xaxes(range = [0, 100])

    # add gridlines on x axis
    fig.update_xaxes(tickmode = 'array', tickvals = [0, 20, 40, 60, 80, 100])
    fig.update_xaxes(showgrid = True, gridwidth = 1, gridcolor = 'grey')
    # get rid of y axis
    fig.update_yaxes(showticklabels = False)
    fig.update_traces(marker_line_width = 1, marker_line_color = 'black')

    return col5.plotly_chart(fig, use_container_width = True)

plot_wing_height_rate()

# drop repeated players, keeping most recent by season
positional_df = positional_df.sort_values(by = 'season', ascending = False)
positional_df = positional_df.drop_duplicates(subset = 'player', keep = 'first')


def plot_height_wingspan():
    # plot height vs wingspan with plotly

    fig = px.scatter(positional_df, x='height_final', y='wingspan_final', hover_name='player',
                    hover_data=['height_final', 'wingspan_final', 'player'])
    fig.update_traces(marker_size=10)
    fig.update_layout(title='Height vs Wingspan for ' + position + 's', width=800, height=500)
    fig.update_xaxes(title='Height (inches)')
    fig.update_yaxes(title='Wingspan (inches)')
    fig.update_traces(marker_line_width=1, marker_line_color='black')
    fig.add_trace(go.Scatter(x=[player_height], y=[player_wingspan], text=[player],
                            mode='markers', marker=dict(size=20, color='red'),
                            hoverinfo='x+y+text', name='Selected Player', showlegend=True))

    # add a line for the average wingspan at the position
    fig.add_trace(go.Scatter(x=[positional_df['height_final'].min(), positional_df['height_final'].max()],
                            y=[positional_df['wingspan_final'].mean(), positional_df['wingspan_final'].mean()],
                            mode='lines', line=dict(color='red', width=1, dash='dash'),
                            name='Average Wingspan', showlegend=True))

    # add a line for the average height at the position
    fig.add_trace(go.Scatter(x=[positional_df['height_final'].mean(), positional_df['height_final'].mean()],
                            y=[positional_df['wingspan_final'].min(), positional_df['wingspan_final'].max()],
                            mode='lines', line=dict(color='red', width=1, dash='dash'),
                            name='Average Height', showlegend=True))

    return st.plotly_chart(fig, use_container_width=True)



# div
st.markdown('---')



def plot_height_wingspan2():
    # plot height vs wingspan with plotly

    fig = px.scatter(positional_df_season_selected, x='height_final', y='wingspan_final', hover_name='player', color='season',
                    hover_data=['height_final', 'wingspan_final', 'player'])
    fig.update_traces(marker_size=10)
    fig.update_layout(title='Height vs Wingspan for ' + position + 's', width=800, height=800)
    fig.update_xaxes(title='Height (inches)')
    fig.update_yaxes(title='Wingspan (inches)')
    fig.update_traces(marker_line_width=1, marker_line_color='black')
    fig.add_trace(go.Scatter(x=[player_height], y=[player_wingspan], text=[player],
                            mode='markers', marker=dict(size=20, color='red'),
                            hoverinfo='x+y+text', name='Selected Player', showlegend=False))

    # add a line for the average wingspan at the position
    fig.add_trace(go.Scatter(x=[positional_df_season_selected['height_final'].min(), positional_df_season_selected['height_final'].max()],
                            y=[positional_df_season_selected['wingspan_final'].mean(), positional_df_season_selected['wingspan_final'].mean()],
                            mode='lines', line=dict(color='red', width=1, dash='dash'),
                            name='Average Wingspan', showlegend=False))

    # add a line for the average height at the position
    fig.add_trace(go.Scatter(x=[positional_df_season_selected['height_final'].mean(), positional_df_season_selected['height_final'].mean()],
                            y=[positional_df_season_selected['wingspan_final'].min(), positional_df_season_selected['wingspan_final'].max()],
                            mode='lines', line=dict(color='red', width=1, dash='dash'),
                            name='Average Height', showlegend=False))

    # encode the png image to base64
    with open(player_photo, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("ascii")

    # add player photo to the plot
    fig.add_layout_image(
        dict(
            source='data:image/png;base64,{}'.format(encoded_image),
            xref="x",
            yref="y",
            x=player_height,
            y=player_wingspan,
            sizex=2,  # adjust image size as needed
            sizey=2,  # adjust image size as needed
            xanchor="center",
            yanchor="middle",
            sizing="contain",
            opacity=1,
            layer="above")
    )

    return st.plotly_chart(fig, use_container_width=True)

pos_seasons = positional_df['season'].unique()
# turn into df
pos_seasons = pd.DataFrame(pos_seasons)
# drop nans
pos_seasons = pos_seasons.dropna()
# make int
pos_seasons = pos_seasons.astype(int)
# turn back into list
pos_seasons = pos_seasons[0].tolist()




# add option to filter by season
seasons = st.multiselect('Season Select', pos_seasons, default = pos_seasons )


# keep only selected seasons
positional_df_season_selected = positional_df[positional_df['season'].isin(seasons)]

plot_height_wingspan2()