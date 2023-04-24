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
    ">NBA Player Game Data</h1>
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


st.markdown("""
    <h2 style="
        font-family: Arial, sans-serif;
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        padding: 10px;
        background: linear-gradient(#24262c, #0e1117);
        box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5); /* Add 3D shadow effect */
        border-radius: 30px;
        border: 1px solid #ffffff;
        width: 80%;
    ">Season Data</h2>
""", unsafe_allow_html=True)

st.write('')

player_gbg_22 = player_gbg[player_gbg['adv_season'] == 2022]


# get season averages
player_ppg = round(player_gbg_22['trad_pts'].sum() / len(player_gbg_22),1)
player_rpg = round(player_gbg_22['trad_reb'].sum() / len(player_gbg_22),1)
player_apg = round(player_gbg_22['trad_ast'].sum() / len(player_gbg_22),1)
player_tovpg = round(player_gbg_22['trad_tov'].sum() / len(player_gbg_22),1)
player_3p_pct = round(player_gbg_22['trad_3pm'].sum() / player_gbg_22['trad_3pa'].sum() *100, 1)
player_fg_pct = round(player_gbg_22['trad_fgm'].sum() / player_gbg_22['trad_fga'].sum() *100, 1)

# display season averages as metrics
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric('Points Per Game', str(round(player_ppg, 2)))
col2.metric('Rebounds Per Game', str(round(player_rpg, 2)))
col3.metric('Assists Per Game', str(round(player_apg, 2)))
col4.metric('Turnovers Per Game', str(round(player_tovpg, 2)))
col5.metric('3 Point %', str(round(player_3p_pct, 2)))
col6.metric('FG %', str(round(player_fg_pct, 2)))

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



st.dataframe(player_gbg.style.format('{:.1f}', subset = numeric_cols))



# calculate averages for player
player_gbg_avg = player_gbg.groupby('player').mean().reset_index()
 
 # show efg% and ts%, 3p%, fg%, adv_pace
player_gbg_avg = player_gbg_avg[['player', 'adv_offrtg', 'adv_defrtg', 'adv_efg%', 'adv_ts%',  'adv_pace', 'ppm']]
ncols = ['adv_offrtg', 'adv_defrtg', 'adv_efg%', 'adv_ts%',  'adv_pace', 'ppm']
# get rid of index
player_gbg_avg = player_gbg_avg.set_index('player')
# rename columns
player_gbg_avg.columns = ['Offensive Rating', 'Defensive Rating', 'Effective FG%', 'True Shooting%', 'Pace', 'Points Per Minute']
st.write('**Advanced Averages**')
st.table(player_gbg_avg.style.format('{:.2f}', subset = player_gbg_avg.columns))

st.write("---")


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


st.markdown("""
    <h2 style="
        font-family: Arial, sans-serif;
        font-size: 30px;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        padding: 20px;
        background: linear-gradient(#24262c, #0e1117);
        box-shadow: 5px 5px 10px rgba(0, 0, 0, 0.5); /* Add 3D shadow effect */
        border-radius: 30px;
    ">Splits: Home vs Away</h2>
""", unsafe_allow_html=True)


# add optional filter for season
season_filter = st.multiselect('Select seasons to filter data by:', player_gbg['adv_season'].unique(), default = player_gbg['adv_season'].unique())

col1, col2, col3, col4, col5 = st.columns([.3, .0333, .3, .0333, .3])
# plot distributions of adv_ts%, 3p%, ppm with plotly

# plot distplot for adv_ts%
colors = ['slategray', 'red']

fig = ff.create_distplot([player_gbg[player_gbg['Home'] == 1]['adv_ts%'], player_gbg[player_gbg['Home'] == 0]['adv_ts%']], ['Home', 'Away'], bin_size = 5,
                            curve_type='normal', # override default 'kde'
                                colors=colors)
fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
                    paper_bgcolor='rgba(0, 0, 0, 0)')

# make colors transparent
fig.data[0].update(opacity=0.2)
fig.data[1].update(opacity=0.2)
fig.update_layout(title = 'TS% Distribution for ' + player)
col1.plotly_chart(fig, use_container_width = True)



# plot distplot for 3p%
colors = ['slategray', 'red']

fig = ff.create_distplot([player_gbg[player_gbg['Home'] == 1]['3p%'], player_gbg[player_gbg['Home'] == 0]['3p%']], ['Home', 'Away'], bin_size = 5,
                            curve_type='normal', # override default 'kde'
                            colors=colors)
fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
                    paper_bgcolor='rgba(0, 0, 0, 0)')
# make colors transparent
fig.data[0].update(opacity=0.2)
fig.data[1].update(opacity=0.2)
fig.update_layout(title = '3P% Distribution for ' + player)
fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
                    paper_bgcolor='rgba(0, 0, 0, 0)')
col3.plotly_chart(fig, use_container_width = True)


# plot distplot for ppm
fig = ff.create_distplot([player_gbg[player_gbg['Home'] == 1]['ppm'], player_gbg[player_gbg['Home'] == 0]['ppm']], ['Home', 'Away'], bin_size = 0.05,
                            curve_type='normal', # override default 'kde'
                                colors=colors)
fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
                    paper_bgcolor='rgba(0, 0, 0, 0)')
# make colors transparent
fig.data[0].update(opacity=0.2)
fig.data[1].update(opacity=0.2)
fig.update_layout(title = 'Points Per Minute Distribution for ' + player)
col5.plotly_chart(fig, use_container_width = True)


c1, c2 = st.columns(2)

# add a plotly ddistplot for player pts at home and away
home_pts = player_gbg[player_gbg['Home'] == 1]['pts']
away_pts = player_gbg[player_gbg['Home'] == 0]['pts']
fig = ff.create_distplot([home_pts, away_pts], ['Home', 'Away'], bin_size = 2, curve_type='normal', colors=colors)
fig.update_layout(title = 'Points Distribution for ' + player + ' at Home and Away*')
# transparent background
fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
                    paper_bgcolor='rgba(0, 0, 0, 0)')
c1.plotly_chart(fig, use_container_width=True)

# add cdf plot for player pts at home and away
fig = px.line(x = np.sort(home_pts), y = np.arange(1, len(home_pts) + 1) / len(home_pts), title = 'CDF of Points for ' + player + ' at Home')
fig.add_scatter(x = np.sort(away_pts), y = np.arange(1, len(away_pts) + 1) / len(away_pts), name = 'Away')
fig.update_layout(title = 'CDF of Points Scored for ' + player + ' at Home and Away')
fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent plot background
                    paper_bgcolor='rgba(0, 0, 0, 0)')
# color background with lower cdf value

c2.plotly_chart(fig, use_container_width=True)

st.markdown('*Note: the CDF (Cumulative Distribution Function) shows in what percentage of games they scored less than or equal to the given number of points. For example, if a player scored 20 points in 10 games, the CDF would show that they scored less than or equal to 20 points in 100% of games. If the Away line is over the Home line, the player scores more at Home.')










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
border-radius: 10px;  /* Adjust this value to change the rounding of corners */
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
