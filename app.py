from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the .pkl model files
with open('ipl1_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('winmodel.pkl', 'rb') as file:
    win_prob_model = pickle.load(file)


# Dictionary to map team names to one-hot encoding
team_mapping = {
    'Chennai Super Kings': [1, 0, 0, 0, 0, 0, 0, 0],
    'Delhi Daredevils': [0, 1, 0, 0, 0, 0, 0, 0],
    'Kings XI Punjab': [0, 0, 1, 0, 0, 0, 0, 0],
    'Kolkata Knight Riders': [0, 0, 0, 1, 0, 0, 0, 0],
    'Mumbai Indians': [0, 0, 0, 0, 1, 0, 0, 0],
    'Rajasthan Royals': [0, 0, 0, 0, 0, 1, 0, 0],
    'Royal Challengers Bangalore': [0, 0, 0, 0, 0, 0, 1, 0],
    'Sunrisers Hyderabad': [0, 0, 0, 0, 0, 0, 0, 1]
}

# Route to render the main HTML page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction endpoint for form 1 (Player Performance)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # Extract form data
    batting_team = data.get("battingTeam")
    bowling_team = data.get("bowlingTeam")
    over = float(data.get("over", 0))
    runs = int(data.get("runs", 0))
    wickets = int(data.get("wickets", 0))
    runs5overs = int(data.get("runs5overs", 0))
    wickets5overs = int(data.get("wickets5overs", 0))

    # Convert team names to one-hot encoding
    batting_team_encoded = team_mapping.get(batting_team, [0] * 8)
    bowling_team_encoded = team_mapping.get(bowling_team, [0] * 8)

    # Prepare data for the model
    model_input = batting_team_encoded + bowling_team_encoded + [runs, wickets, over, runs5overs, wickets5overs]
    model_input = np.array([model_input])  # Reshape input for prediction

    # Make prediction
    prediction = model.predict(model_input)

    # Return the prediction to the frontend
    return jsonify({"prediction": int(round(prediction[0]))})

# Prediction endpoint for form 2 (Win Probability)
@app.route('/predict_win_probability', methods=['POST'])
def predict_win_probability():
    data = request.form
    batting_team = data.get("battingTeam")
    bowling_team = data.get("bowlingTeam")
    city = data.get("city")
    runs_left = int(data.get("runsleft", 0))
    balls_left = int(data.get("ballsleft", 0))
    wickets_left = int(data.get("wicketsleft", 0))
    currrr = float(data.get("currrr", 0))
    reqrr = float(data.get("reqrr", 0))
    target = int(data.get("target", 0))

    l = [[batting_team,bowling_team, city, runs_left, balls_left,wickets_left ,currrr, reqrr, target]]
    columns = ['BattingTeam', 'BowlingTeam', 'City', 'runs_left', 'balls_left',
       'wickets_left', 'current_run_rate', 'required_run_rate', 'target']

    team2023 = pd.DataFrame(l, columns=columns)

    prob = win_prob_model.predict_proba(team2023)[0][1]  # Example if you want the second class probability
    return jsonify({"prediction": float(prob)})  # Return a float for readability


if __name__ == '__main__':
    app.run(debug=True)
