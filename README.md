# HUMAN-DISEASE-PREDICTION
Application Development-1
from flask import Flask, request, render_template_string, redirect, url_for, flash, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import Counter
from fuzzywuzzy import process
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load and prepare data
DATA_PATH = "DiseaseAndSymptoms1.csv"  # Update with your dataset path
data = pd.read_csv(DATA_PATH).dropna(axis=1)

# Prepare features and labels
X = data.iloc[:, 1:]  # Symptoms
y = data.iloc[:, 0]    # Disease

# Encode labels for the target
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# One-Hot Encoding for categorical features
one_hot_encoder = OneHotEncoder(sparse_output=False)
X_encoded = one_hot_encoder.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=24)

# Train models
models = {
    'Random Forest': RandomForestClassifier(random_state=18),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC()
}
for model_name, model in models.items():
    model.fit(X_train, y_train)

# Calculate accuracy for each model
accuracy_results = {name: accuracy_score(y_test, model.predict(X_test)) * 100 for name, model in models.items()}

def predict_disease(symptoms):
    input_data = [0] * len(one_hot_encoder.get_feature_names_out())
    symptom_index = {symptom: index for index, symptom in enumerate(one_hot_encoder.get_feature_names_out())}
    
    invalid_symptoms = []

    for symptom in symptoms:
        symptom = symptom.strip().lower().replace('_', ' ')
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1
        else:
            closest_match, score = process.extractOne(symptom, symptom_index.keys())
            if score >= 80:
                input_data[symptom_index[closest_match]] = 1
            else:
                invalid_symptoms.append(symptom)

    input_data = np.array(input_data).reshape(1, -1)

    predictions = {}
    if any(input_data[0]):
        for model_name, model in models.items():
            prediction_raw = model.predict(input_data)
            prediction = encoder.inverse_transform(prediction_raw)[0]
            predictions[model_name] = prediction
        final_prediction = Counter(predictions.values()).most_common(1)[0][0]
        predictions['Final Prediction'] = final_prediction
        predictions['Model Accuracies'] = accuracy_results  # Keep accuracies here
    else:
        predictions = {"Error": "No valid symptoms provided.", "Invalid Symptoms": invalid_symptoms}

    return predictions

users = {}

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and check_password_hash(users[username], password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
    return render_template_string(LOGIN_HTML)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        if username in users:
            flash('Username already exists. Please choose another.', 'danger')
        else:
            users[username] = password
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template_string(REGISTRATION_HTML)

@app.route('/index', methods=['GET', 'POST'])
def index():
    predictions = {}
    if request.method == 'POST':
        symptoms_input = [
            request.form['symptom1'],
            request.form['symptom2'],
            request.form['symptom3'],
            request.form['symptom4'],
            request.form['symptom5']
        ]
        predictions = predict_disease(symptoms_input)

    return render_template_string(INDEX_HTML, predictions=predictions)

# HTML Templates with Enhanced Styling
LOGIN_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login</title>
    <style>
        body { background: #f4f4f4; font-family: 'Segoe UI', sans-serif; }
        .container { background: #ffffff; padding: 40px; border-radius: 12px; box-shadow: 0 0 25px rgba(0, 0, 0, 0.1); width: 400px; margin: 50px auto; }
        h1 { color: #333; border-bottom: 2px solid #28a745; padding-bottom: 10px; }
        input[type="text"], input[type="password"] { width: 100%; padding: 10px; margin-bottom: 15px; border: 1px solid #ddd; border-radius: 6px; }
        input[type="submit"] { background: #28a745; color: white; border: none; padding: 10px; cursor: pointer; border-radius: 6px; }
        input[type="submit"]:hover { background: #218838; }
        a { color: #28a745; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .message { margin-top: 10px; }
        .reset-btn { margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Login</h1>
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required autocomplete="off">
            <input type="password" name="password" placeholder="Password" required autocomplete="off">
            <input type="submit" value="Login">
            <input type="button" value="Reset" onclick="document.querySelector('form').reset();">
        </form>
        <div class="message">
            <a href="{{ url_for('register') }}">Not registered? Sign up here.</a>
        </div>
    </div>
</body>
</html>
"""

REGISTRATION_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Register</title>
    <style>
        body { background: #f4f4f4; font-family: 'Segoe UI', sans-serif; }
        .container { background: #ffffff; padding: 40px; border-radius: 12px; box-shadow: 0 0 25px rgba(0, 0, 0, 0.1); width: 400px; margin: 50px auto; }
        h1 { color: #333; border-bottom: 2px solid #28a745; padding-bottom: 10px; }
        input[type="text"], input[type="password"] { width: 100%; padding: 10px; margin-bottom: 15px; border: 1px solid #ddd; border-radius: 6px; }
        input[type="submit"] { background: #28a745; color: white; border: none; padding: 10px; cursor: pointer; border-radius: 6px; }
        input[type="submit"]:hover { background: #218838; }
        .reset-btn { margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Register</h1>
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required autocomplete="off">
            <input type="password" name="password" placeholder="Password" required autocomplete="off">
            <input type="submit" value="Register">
            <input type="button" value="Reset" onclick="document.querySelector('form').reset();">
        </form>
    </div>
</body>
</html>
"""

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Disease Prediction</title>
    <style>
        body { background: #f4f4f4; font-family: 'Segoe UI', sans-serif; }
        .container { background: #ffffff; padding: 40px; border-radius: 12px; box-shadow: 0 0 25px rgba(0, 0, 0, 0.1); width: 600px; margin: 50px auto; }
        h1 { color: #333; border-bottom: 2px solid #28a745; padding-bottom: 10px; }
        input[type="text"] { width: calc(50% - 10px); padding: 10px; margin: 5px; border: 1px solid #ddd; border-radius: 6px; }
        input[type="submit"] { background: #28a745; color: white; border: none; padding: 10px; cursor: pointer; border-radius: 6px; }
        input[type="submit"]:hover { background: #218838; }
        .results { margin-top: 20px; }
        .result-box { background: #e9ecef; padding: 15px; margin-bottom: 15px; border-radius: 8px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); }
        .final-prediction { background: #d4edda; border-left: 5px solid #28a745; }
        .reset-btn { margin-top: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Disease Prediction</h1>
        <form method="POST">
            <input type="text" name="symptom1" placeholder="Symptom 1" required>
            <input type="text" name="symptom2" placeholder="Symptom 2">
            <input type="text" name="symptom3" placeholder="Symptom 3">
            <input type="text" name="symptom4" placeholder="Symptom 4">
            <input type="text" name="symptom5" placeholder="Symptom 5">
            <input type="submit" value="Predict">
            <input type="button" value="Reset" onclick="document.querySelector('form').reset();">
        </form>
        <div class="results">
            {% if predictions %}
                {% for key, value in predictions.items() %}
                    <div class="result-box">
                        <strong>{{ key }}:</strong> 
                        {% if key == 'Model Accuracies' %}
                            <ul>
                            {% for model, accuracy in value.items() %}
                                <li>{{ model }}: {{ accuracy | round(2) }}%</li>
                            {% endfor %}
                            </ul>
                        {% else %}
                            {{ value }}
                        {% endif %}
                    </div>
                {% endfor %}
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)
