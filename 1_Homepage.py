import streamlit as st
import pandas as pd
import numpy as np
import datetime as datetime

# Title
st.title('NBA Player Analyzer')
st.write('Please select a team, player, and position to get started!')


# get current time in pst
pst = datetime.timezone(datetime.timedelta(hours=-8))
# to datetime
pst = datetime.datetime.now(pst)
# assign today value
today = pst.strftime('%Y-%m-%d')

# Load Data
player_numbers = pd.read_csv('data/player/nba_com_info/players_and_photo_links.csv')

# add capitalized player name TODO: THIS MESSES UP SOME PLAYER NAMES WITH NON-TITLE RULES
player_numbers['Player'] = player_numbers['player_name'].str.title()

# Load Sizes
df_sizes = pd.read_csv('data/player/aggregates_of_aggregates/New_Sizes_and_Positions.csv')

# load game by game data
gbg_df = pd.read_csv('data/player/aggregates/Trad&Adv_box_scores_GameView.csv')

# select team
teams = gbg_df['trad_team'].unique()

# sort teams
teams = np.sort(teams)

cols = st.columns(2)

# select team
st.subheader('Select a Team')
team = st.selectbox('', teams, index = 24)

# select player
gbg_22 = gbg_df[gbg_df['adv_season'] == 2022]

players_22 = gbg_22[gbg_22['trad_team'] == team]['trad_player'].unique()

# sort players
players_22 = np.sort(players_22)

st.subheader('Select a Player')

#create select box for player
player = st.selectbox('', players_22, index = 2)


#### Assign Session States ####
st.session_state['player'] = player
st.session_state['team'] = team

# assign team num, which is index of team in teams. Used to call team data on other pages.
team_num = np.where(teams == team)[0][0]
st.session_state['team_num'] = team_num

# player number is index
player_number = np.where(players_22 == player)[0][0]

# Assign Session State for player number
st.session_state['player_number'] = player_number


# get player nba id
player_nba_id = player_numbers[player_numbers['Player'] == player]['nba_id'].iloc[0]

# get player photo
player_photo = 'data/player/photos/photos/' + str(player_nba_id) + '.png'

#set
st.session_state['player_photo'] = player_photo

# add player photo
st.image(player_photo, width = 300)

# select position
position_options = ['PG', 'SG', 'SF', 'PF', 'C']

st.subheader('Select a Position')

# 
position = st.selectbox('Select Position to evaluate the player at', position_options)

# Assign Session States
st.session_state['position'] = position

#get index of position in position_options, 0 if PG, 1 if SG, etc.
position_index_dict = {'PG': 0, 'SG': 1, 'SF': 2, 'PF': 3, 'C': 4}
position_index = position_index_dict[position]

st.session_state['position_index'] = position_index

##### FINISHED SESSION STATES #####


# Write session states out on sidebar
# st.sidebar.write('Session States!')
# st.sidebar.write('')
# st.sidebar.write('Player: ' + st.session_state['player'])
# st.sidebar.write('Player Number: ' + str(st.session_state['player_number']))
# st.sidebar.write('Team: ' + st.session_state['team'])
# st.sidebar.write('Team Number: ' + str(st.session_state['team_num']))
# st.sidebar.write('Position: ' + st.session_state['position'])
# st.sidebar.write('Position Index: ' + str(st.session_state['position_index']))



st.write('''

''')

st.subheader('Select from the left menu to move through the tools')


st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('Contact: [LinkedIn](https://www.linkedin.com/in/travis-royce/) | [GitHub](https://github.com/tmcroyce) | traviscroyce@gmail.com')

