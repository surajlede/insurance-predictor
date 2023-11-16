from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Linear Regression model
loc = r"linear_regression_model.pkl"
with open(loc, 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        if request.form['region'] == 'Southwest':
            ne,nw,se,sw = 0,0,0,1
        elif request.form['region'] == 'Southeast':
            ne,nw,se,sw = 0,0,1,0
        elif request.form['region'] == 'Northwest':
            ne,nw,se,sw = 0,1,0,0    
        elif request.form['region'] == 'Northeast':
            ne,nw,se,sw = 1,0,0,0

        # Convert the features to a NumPy array and make predictions
        features = np.array([age, sex, bmi, children, smoker, ne,nw,se,sw]).reshape(1, -1)
        predictions = model.predict(features)

        # Return the predictions as JSON
        result = np.exp(predictions[0])
        return f"Insurance Prediction = $ {np.round(result, 2)}"
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")
