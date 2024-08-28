from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the saved Random Forest model
rf_model = joblib.load('rf_model.joblib')

app = Flask(__name__)

# Define the selected features
selected_features = ['sweating', 'trouble.in.concentration', 'trouble.sleeping', 'close.friend', 'introvert',
                     'popping.up.stressful.memory', 'avoids.people.or.activities', 'feeling.negative',
                     'blamming.yourself', 'increased.energy']

meaningful_names = {
    'sweating': 'Sweating',
    'trouble.in.concentration': 'Trouble in Concentration',
    'trouble.sleeping': 'Trouble Sleeping',
    'close.friend': 'Close Friend',
    'introvert': 'Introvert',
    'popping.up.stressful.memory': 'Stressful Memory',
    'avoids.people.or.activities': 'Avoids People or Activities',
    'feeling.negative': 'Feeling Negative',
    'blamming.yourself': 'Blaming Yourself',
    'increased.energy': 'Increased Energy',
}

@app.route('/')
def home():
    return render_template('index.html', selected_features=selected_features)

@app.route('/predict', methods=['POST'])
def predict():
    input_values = []
    
    for feature in selected_features:
        form_value = request.form[feature].lower()
        if form_value == 'yes':
            input_values.append(1)
        elif form_value == 'no':
            input_values.append(0)
        else:
            return render_template('error.html', error_message=f"Invalid value for {feature}")

     # Check if all values are 0
    if sum(input_values) == 0:
        return render_template('no_disorder.html')

    prediction = rf_model.predict(np.array([input_values]))[0]
    predicted_disorder = meaningful_names.get(prediction, prediction)

    return render_template('result.html', predicted_disorder=predicted_disorder)

if __name__ == '__main__':
    app.run(debug=True)