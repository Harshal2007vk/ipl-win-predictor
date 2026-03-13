import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

st.title("🏏 IPL Win Predictor")
st.write("App loaging successfully plzz wait guys for 2-3 min ")
st.markdown("""
### About Me

Hi, I’m **Harshal Kapale** 👋, currently a **2nd semester student** exploring **Data Science and Machine Learning**.

This project is an **IPL Win Predictor** where I cleaned the dataset, trained a **machine learning model**, and built an interactive web app.

I'm learning **Python, NumPy, Pandas, Matplotlib, Seaborn, Plotly, Kaggle, and Git/GitHub**.
""")
# Load dataset
df = pd.read_csv("IPL_processed_dataset.csv")

# Features
X = df.drop(columns=['result'])
y = df['result']

categorical_cols = ['batting_team','bowling_team','city']

transformer = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), categorical_cols)
    ],
    remainder='passthrough'
)

model = RandomForestClassifier()

pipe = Pipeline(steps=[
    ('step1', transformer),
    ('step2', model)
])

pipe.fit(X,y)

# UI
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
wickets = st.number_input("Wickets Left",0,10)
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

    result = pipe.predict(input_df)

    if result[0] == 1:
        st.success(f"{batting_team} will win 🏆")
    else:
        st.error(f"{bowling_team} will win 🏆")