import streamlit as st
import pandas as pd
import numpy as np
import datetime as datetime
import unidecode
import re


st.set_page_config(page_title='Homepage', page_icon=None, layout="wide", initial_sidebar_state="auto" )

title_color_1 = '#282229' # dark purple
title_color_2 = '#0e1117' # dark blue
background_color_1 = '#2c3333' # dark grey
background_color_2 = '#35363C' # dark grey

# set session states for color pallette
st.session_state['title_color_1'] = title_color_1
st.session_state['title_color_2'] = title_color_2
st.session_state['background_color_1'] = background_color_1
st.session_state['background_color_2'] = background_color_2

st.markdown(f"""
    <h1 style="
        font-family: Arial, sans-serif;
        font-size: 48px;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        padding: 20px;
        background: linear-gradient(to right, {title_color_1}, {title_color_2});
        box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5); /* Add 3D shadow effect */
        border-radius: 30px;
    ">NBA Player Analytics Dashboard</h1>
""", unsafe_allow_html=True)

st.write('')


st.markdown("""
<h2 style="
        font-family: Arial, sans-serif;
        font-size: 20px;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        padding: 10px;
        box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5); /* Add 3D shadow effect */
        border-radius: 30px;;
    ">Use this interactive dashboard to analyze and evaluate NBA players by position. Start by selecting a team, player, and position</h2>
    
""", unsafe_allow_html=True)


# get current time in pst
pst = datetime.timezone(datetime.timedelta(hours=-8))
# to datetime
pst = datetime.datetime.now(pst)
# assign today value
today = pst.strftime('%Y-%m-%d')

# Load Data
player_numbers = pd.read_csv('data/player/nba_com_info/players_and_photo_links.csv')

# add capitalized player name column
player_numbers['Player'] = player_numbers['player_name'].str.title()

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

player_numbers['Player'] = player_numbers['Player'].apply(clean_name)


# Load Sizes
df_sizes = pd.read_csv('data/player/aggregates_of_aggregates/New_Sizes_and_Positions.csv')

# load game by game data
gbg_df = pd.read_csv('data/player/aggregates/Trad&Adv_box_scores_GameView.csv')

# fix fighter names
gbg_df['trad_player'] = gbg_df['trad_player'].apply(clean_name)

# select team
teams = gbg_df['trad_team'].unique()

# sort teams
teams = np.sort(teams)
cols = st.columns(2)


# Row 1: Select team
col1, col2, col3, col4 = st.columns([.2, .3, .3, .2])
with col2:
    st.subheader(' ')
    st.subheader('Select a Team')
with col3:
    team = st.selectbox('', teams, index=24, key='team_select')

# Select player
gbg_22 = gbg_df[gbg_df['adv_season'] == 2022]
players_22 = gbg_22[gbg_22['trad_team'] == team]['trad_player'].unique()
# Sort players
players_22 = np.sort(players_22)

# Row 2: Select player
col1, col2, col3, col4 = st.columns([.2, .3, .3, .2])
with col2:
    st.subheader(' ')
    st.subheader('Select a Player')
with col3:
    player = st.selectbox('', players_22, index=3, key='player_select')

# Select position
position_options = ['PG', 'SG', 'SF', 'PF', 'C']

# Row 3: Select position
col1, col2, col3, col4 = st.columns([.2, .3, .3, .2])
with col2:
    st.subheader(' ')
    st.subheader('Select a Position')
with col3:
    position = st.selectbox('', position_options, key='position_select')

st.write(' ')
st.write(' ')

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

# add selected player name
st.sidebar.subheader('Selected Player')
st.sidebar.write(player)

# add player photo
st.sidebar.image(player_photo, width = 300)


# st.subheader('Select a Position')
# position = st.selectbox('Select Position to evaluate the player at', position_options)

# Assign Session States
st.session_state['position'] = position

#get index of position in position_options, 0 if PG, 1 if SG, etc.
position_index_dict = {'PG': 0, 'SG': 1, 'SF': 2, 'PF': 3, 'C': 4}
position_index = position_index_dict[position]

st.session_state['position_index'] = position_index


st.write('''



''')


st.markdown("""
<h2 style="
        font-family: Arial, sans-serif;
        font-size: 20px;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        padding: 10px;
        box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5); /* Add 3D shadow effect */
        border-radius: 30px;;
    ">...Then, select from the left menu to move through the tools</h2>
    
""", unsafe_allow_html=True)


st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')

# Use markdown for the footer and contact information
st.sidebar.markdown("""
## Contact Information

If you have any questions or feedback, feel free to reach out:

- [LinkedIn](https://www.linkedin.com/in/travis-royce/)
- [GitHub](https://github.com/tmcroyce)
- Email: traviscroyce@gmail.com

Thanks for using the NBA Player Analytics Toolkit!
""", unsafe_allow_html=True)




############



# Total
custom_background = """
<style>
[data-testid="stAppViewContainer"] {
background: linear-gradient(to right, #2c3333, #35363C);
}
</style>
"""
st.markdown(custom_background, unsafe_allow_html=True)


# Define custom CSS for the gradient background
custom_header = """
<style>
[data-testid="stHeader"] {
background: linear-gradient(to right, #2c3333, #35363C);
}
</style>
"""

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



custom_sidebar = """
<style>
section[data-testid="stSidebar"]{
background-image: linear-gradient(#35363C, #0e1117);
color: white;
}
</style>
"""
st.markdown(custom_sidebar , unsafe_allow_html=True)


custom_selectbox = """
<style>
[data-baseweb="select"]{
background-image: linear-gradient(#35363C, #0e1117);
border-radius: 30px;  /* Adjust this value to change the rounding of corners */
text-align: center;  /* Center the text inside the metric box */
color: white;
}
</style>
"""
st.markdown(custom_selectbox , unsafe_allow_html=True)

