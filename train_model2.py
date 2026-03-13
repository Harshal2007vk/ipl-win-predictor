import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(r"C:\Users\A15\Desktop\ML_project1\IPL_processed_dataset.csv")


print(df.head())

# Features
X = df.drop(columns=['result'])

# Target
y = df['result']

# Categorical columns
categorical_cols = ['batting_team','bowling_team','city']

# Preprocessing
transformer = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), categorical_cols)
    ],
    remainder='passthrough'
)

# Model
model = RandomForestClassifier()

# Pipeline
pipe = Pipeline(steps=[
    ('step1', transformer),
    ('step2', model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
pipe.fit(X_train, y_train)

# Predict
y_pred = pipe.predict(X_test)

# Accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))



pickle.dump(pipe, open("ipl_win_predictor.pkl","wb"))
print("Model saved successfully")