from flask import Flask, render_template, request, jsonify
from random_sample import create_sample_files
import random
import pandas as pd
app = Flask(__name__)

# Mock sample file paths (replace with actual logic from random_sample.py)
SAMPLE_FILE_PATHS = [f"sample_{i}.csv" for i in range(1, 6)]

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
    num_records_per_sample = 1000
    create_sample_files(original_data, num_samples, num_records_per_sample)
    return "New samples generated!"  # Placeholder message

@app.route("/predict")
def predict():
    selected_file = request.args.get("selected_file")
    if selected_file:
        try:
            # Read selected file into a pandas DataFrame
            sample_df = pd.read_csv(selected_file)
            # Save DataFrame as predict_data.csv
            sample_df.to_csv("predict_data.csv", index=False)
            return jsonify({"success": True})
        except FileNotFoundError:
            return jsonify({"success": False, "error": "File not found"})
    else:
        return jsonify({"success": False, "error": "No file selected"})

if __name__ == "__main__":
    app.run(debug=True)
