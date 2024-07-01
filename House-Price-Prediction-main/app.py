from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

# Load the saved model
random_forest_model = joblib.load('random_forest_model.joblib')

# Initialize Flask application
app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return "Hello world"

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        print('Received data:', data)

        # Prepare input data
        input_data = {
            'POSTED_BY': data['POSTED_BY'],
            'UNDER_CONSTRUCTION': data['UNDER_CONSTRUCTION'],
            'RERA': data['RERA'],
            'BHK_NO': data['BHK_NO'],
            'BHK_OR_RK': data['BHK_OR_RK'],
            'SQUARE_FT': data['SQUARE_FT'],
            'READY_TO_MOVE': data['READY_TO_MOVE'],
            'RESALE': data['RESALE'],
            'LONGITUDE': data['LONGITUDE'],
            'LATITUDE': data['LATITUDE']
        }

        # Convert the input data to a 2D array
        input_array = np.array([list(input_data.values())])

        # Make predictions using the loaded model
        predictions = random_forest_model.predict(input_array)

        # Return prediction as JSON response
        return jsonify({'prediction': predictions[0]}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
