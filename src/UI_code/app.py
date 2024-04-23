from flask import Flask, render_template, request, jsonify
import random
import os
import pandas as pd
import requests
import json

app = Flask(__name__)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# sample file paths
SAMPLE_FILE_PATHS = [f"sample_{i}.csv" for i in range(1, 6)]
predict_url  = 'http://127.0.0.1:8080/predict'

# Defining a function to create sample files
def create_sample_files(original_data, num_samples, num_records_per_sample):
    for i in range(num_samples):
        sample_data = original_data.sample(n=num_records_per_sample)
        '''SAMPLE_PATH = os.path.join(PROJECT_DIR, 'data','sample_test_data_{i+1}.csv')
        sample_data.to_csv(SAMPLE_PATH, index=False)'''
        sample_data.to_csv(f'sample_{i+1}.csv', index=False)
        print(f'Sample test data {i+1} created successfully.')

'''
This module creates the UI screen to select a sample file and predict
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
            # Reading the selected file into a pandas DataFrame
            sample_df = pd.read_csv(selected_file)
            sample_data = sample_df.head(5).values.tolist()  # Converting DataFrame head to list of lists
            column_labels = sample_df.columns.tolist()  # Getting all the column labels
        except FileNotFoundError:
            # Handling file not found case
            pass

    selected_file_options = SAMPLE_FILE_PATHS  # Listing all sample files for the dropdown menu

    return render_template("index.html", selected_file=selected_file, sample_data=sample_data, column_labels=column_labels, selected_file_options=selected_file_options)

@app.route("/generate_new_samples")
def generate_new_samples():
    original_data = pd.read_csv("test_data.csv")
    num_samples = 5
    num_records_per_sample = 4
    create_sample_files(original_data, num_samples, num_records_per_sample)
    return "New samples generated!" 

@app.route("/predict")
def predict():
    selected_file = request.args.get("selected_file")
    if selected_file:
        try:
            # Reading the selected file into a pandas DataFrame
            sample_df = pd.read_csv(selected_file)
            # Saving DataFrame as predict_data.json
            json_data = sample_df.to_json(orient="records")
            data = {
              "instances": json_data
            }
            json_file_path = "predict_data.json"
            #print(data)
            data_json = jsonify(data)            
            with open(json_file_path, "w") as json_file:
                json_file.write(json_data)
        
            return jsonify({"success": True})
        except FileNotFoundError:
            return jsonify({"success": False, "error": "File not found"})
    else:
        return jsonify({"success": False, "error": "No file selected"})

@app.route("/get_response")
def get_response():
    try:
        # Reading the saved JSON file
        with open("predict_data.json", "r") as json_file:
            json_data = json.load(json_file)
            data = {
              "instances": json_data
            }
        
        # Sending a POST request to the prediction endpoint
        response = requests.post(predict_url, json=data)
        
        # Extracting the predictions from the predict.py
        predictions = response.json()["predictions"]
        
        # Returning the predictions for displaying on UI
        return jsonify({"success": True, "predictions": predictions})
    except FileNotFoundError:
        return jsonify({"success": False, "error": "JSON file not found"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    original_data = pd.read_csv("test_data.csv")
    num_samples = 5
    num_records_per_sample = 4
    create_sample_files(original_data, num_samples, num_records_per_sample)
    app.run(debug=True)
