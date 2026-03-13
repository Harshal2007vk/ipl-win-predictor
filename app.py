import streamlit as st
import pickle
import pandas as pd

# Load trained model
model = pickle.load(open("ipl_win_predictor.pkl","rb"))

st.title("IPL Win Predictor")

batting_team = st.selectbox(
"Batting Team",
['Chennai Super Kings','Mumbai Indians','Royal Challengers Bengaluru',
 'Kolkata Knight Riders','Delhi Capitals','Punjab Kings',
 'Rajasthan Royals','Sunrisers Hyderabad']
)

bowling_team = st.selectbox(
"Bowling Team",
['Chennai Super Kings','Mumbai Indians','Royal Challengers Bengaluru',
 'Kolkata Knight Riders','Delhi Capitals','Punjab Kings',
 'Rajasthan Royals','Sunrisers Hyderabad']
)

city = st.selectbox(
"City",
['Mumbai','Delhi','Bangalore','Chennai','Hyderabad','Kolkata']
)

runs_left = st.number_input("Runs Left")
balls_left = st.number_input("Balls Left")
wickets = st.number_input("Wickets Left")
runs_target = st.number_input("Target")
crr = st.number_input("Current Run Rate")
rrr = st.number_input("Required Run Rate")

if st.button("Predict Winner"):

    input_df = pd.DataFrame({
        'batting_team':[batting_team],
        'bowling_team':[bowling_team],
        'city':[city],
        'runs_left':[runs_left],
        'balls_left':[balls_left],
        'wickets_remaining':[wickets],
        'runs_target':[runs_target],
        'crr':[crr],
        'rrr':[rrr]
    })

    result = model.predict(input_df)

    if result[0] == 1:
        st.success(f"{batting_team} will win 🏏")
    else:
        st.error(f"{bowling_team} will win 🏏")