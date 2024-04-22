from flask import Flask, render_template, request, jsonify
#from random_sample import create_sample_files
import random
import os
import pandas as pd
app = Flask(__name__)
import requests

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mock sample file paths (replace with actual logic from random_sample.py)
SAMPLE_FILE_PATHS = [f"sample_{i}.csv" for i in range(1, 6)]
predict_url  = 'http://127.0.0.1:8080/predict'

# Define function to create sample files
def create_sample_files(original_data, num_samples, num_records_per_sample):
    for i in range(num_samples):
        sample_data = original_data.sample(n=num_records_per_sample)
        '''SAMPLE_PATH = os.path.join(PROJECT_DIR, 'data','sample_test_data_{i+1}.csv')
        sample_data.to_csv(SAMPLE_PATH, index=False)'''
        sample_data.to_csv(f'sample_{i+1}.csv', index=False)
        print(f'Sample test data {i+1} created successfully.')

'''
This module creates the UI screen to select a predict file
'''
@app.route("/")
def index():
    selected_file = request.args.get("selected_file")
    if not selected_file:
        selected_file = random.choice(SAMPLE_FILE_PATHS)

    sample_data = None
    column_labels = None
    if selected_file:
        try:
            # Read selected file into a pandas DataFrame
            sample_df = pd.read_csv(selected_file)
            sample_data = sample_df.head(5).values.tolist()  # Convert DataFrame head to list of lists
            column_labels = sample_df.columns.tolist()  # Get column labels
        except FileNotFoundError:
            # Handle file not found case (optional)
            pass

    selected_file_options = SAMPLE_FILE_PATHS  # List of all sample files for dropdown

    return render_template("index.html", selected_file=selected_file, sample_data=sample_data, column_labels=column_labels, selected_file_options=selected_file_options)

@app.route("/generate_new_samples")
def generate_new_samples():
    # Call your random_sample.py function to generate new samples (implementation details not provided)
    # Update SAMPLE_FILE_PATHS accordingly
    original_data = pd.read_csv("test_data.csv")
    # Define parameters
    num_samples = 5
    num_records_per_sample = 4
    create_sample_files(original_data, num_samples, num_records_per_sample)
    return "New samples generated!"  # Placeholder message

@app.route("/predict", methods=["POST"])
def predict():
    selected_file = request.form.get("selected_file")  # Get the selected sample file
    if selected_file:
        try:
            # Read selected file into a pandas DataFrame
            sample_df = pd.read_csv(selected_file)
            # Convert DataFrame to JSON
            json_data = sample_df.to_json(orient="records")
            
            # Send a POST request to predict.py
            response = requests.post(predict_url, json={"instances": json.loads(json_data)})
            if response.status_code == 200:
                # If successful, display predictions on the webpage
                prediction_data = response.json()
                predictions = prediction_data["predictions"]
                return render_template("index.html", selected_file=selected_file, sample_data=sample_data, column_labels=column_labels, selected_file_options=selected_file_options, predictions=predictions)
            else:
                # If request fails, handle error
                return jsonify({"success": False, "error": "Prediction failed"})
        except FileNotFoundError:
            return jsonify({"success": False, "error": "File not found"})
    else:
        return jsonify({"success": False, "error": "No file selected"})

'''@app.route("/predict")
def predict():
    selected_file = request.args.get("selected_file")
    if selected_file:
        try:
            # Read selected file into a pandas DataFrame
            sample_df = pd.read_csv(selected_file)
            # Save DataFrame as predict_data.csv
            json_data = sample_df.to_json(orient="records")
            data = {
              "instances": json_data
            }
            json_file_path = "predict_data.json"
            print(data["instances"])
            data_json = jsonify(data)            
            with open(json_file_path, "w") as json_file:
                json_file.write(json_data)
            
            #response_predict = requests.post(predict_url, json=data)
            #predict_status = response_predict.json()
            #sample_df.to_csv("predict_data.csv", index=False)
            return jsonify({"success": True})
        except FileNotFoundError:
            return jsonify({"success": False, "error": "File not found"})
    else:
        return jsonify({"success": False, "error": "No file selected"})'''

if __name__ == "__main__":
    original_data = pd.read_csv("test_data.csv")
    # Define parameters
    num_samples = 5
    num_records_per_sample = 4
    create_sample_files(original_data, num_samples, num_records_per_sample)
    app.run(debug=True)

