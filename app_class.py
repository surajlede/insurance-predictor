from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

class InsurancePredictor:
    def __init__(self, model_file_location):
        self.app = Flask(__name__)  # Create the Flask app here

        # Load the trained Linear Regression model
        with open(model_file_location, 'rb') as model_file:
            self.model = pickle.load(model_file)

        # Define routes within the constructor
        @self.app.route('/')
        def index():
            return render_template('predict.html')

        @self.app.route('/predict', methods=['POST'])
        def predict():
            try:
                # Extract form data
                age = float(request.form['age'])
                sex = int(request.form['sex'])
                bmi = float(request.form['bmi'])
                children = int(request.form['children'])
                smoker = int(request.form['smoker'])
                region = request.form['region']

                ne, nw, se, sw = self.process_region(region)

                # Convert the features to a NumPy array and make predictions
                features = np.array([age, sex, bmi, children, smoker, ne, nw, se, sw]).reshape(1, -1)
                predictions = self.model.predict(features)

                # Return the predictions as JSON
                result = np.exp(predictions[0])
                return f"Insurance Prediction = $ {np.round(result, 2)}"
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    def process_region(self, region):
        ne, nw, se, sw = 0, 0, 0, 0
        if region == 'Southwest':
            sw = 1
        elif region == 'Southeast':
            se = 1
        elif region == 'Northwest':
            nw = 1
        elif region == 'Northeast':
            ne = 1
        return ne, nw, se, sw

if __name__ == '__main__':
    model_file_location = r"linear_regression_model.pkl"
    predictor = InsurancePredictor(model_file_location)
    predictor.app.run(debug=False, host="0.0.0.0")
