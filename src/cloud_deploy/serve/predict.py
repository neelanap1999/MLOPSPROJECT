from flask import Flask, jsonify, request
from google.cloud import storage
import joblib
import os
import json
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

def initialize_variables():
    """
    Initialize environment variables.
    Returns:
        tuple: The project id and bucket name.
    """
    project_id = os.getenv("PROJECT_ID")
    bucket_name = os.getenv("BUCKET_NAME")
    return project_id, bucket_name

def initialize_client_and_bucket(bucket_name):
    """
    Initialize a storage client and get a bucket object.
    Args:
        bucket_name (str): The name of the bucket.
    Returns:
        tuple: The storage client and bucket object.
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    return storage_client, bucket

def load_stats(bucket, SCALER_BLOB_NAME='scaler/normalization_stats.json'):
    """
    Load normalization stats from a blob in the bucket.
    Args:
        bucket (Bucket): The bucket object.
        SCALER_BLOB_NAME (str): The name of the blob containing the stats.
    Returns:
        dict: The loaded stats.
    """
    scaler_blob = bucket.blob(SCALER_BLOB_NAME)
    stats_str = scaler_blob.download_as_text()
    stats = json.loads(stats_str)
    return stats

def load_model(bucket, bucket_name):
    """
    Fetch and load the latest model from the bucket.
    Args:
        bucket (Bucket): The bucket object.
        bucket_name (str): The name of the bucket.
    Returns:
        _BaseEstimator: The loaded model.
    """
    latest_model_blob_name = fetch_latest_model(bucket_name)
    local_model_file_name = os.path.basename(latest_model_blob_name)
    model_blob = bucket.blob(latest_model_blob_name)
    model_blob.download_to_filename(local_model_file_name)
    model = joblib.load(local_model_file_name)
    return model

def fetch_latest_model(bucket_name, prefix="model/model_"):
    """Fetches the latest model file from the specified GCS bucket.
    Args:
        bucket_name (str): The name of the GCS bucket.
        prefix (str): The prefix of the model files in the bucket.
    Returns:
        str: The name of the latest model file.
    """
    # List all blobs in the bucket with the given prefix
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    # Extract the timestamps from the blob names and identify the blob with the latest timestamp
    blob_names = [blob.name for blob in blobs]
    if not blob_names:
        raise ValueError("No model files found in the GCS bucket.")

    latest_blob_name = sorted(blob_names, key=lambda x: x.split('_')[-1], reverse=True)[0]

    return latest_blob_name


def normalize_data(instance, stats):
    """
    Normalizes a data instance using provided statistics.
    Args:
        instance (dict): A dictionary representing the data instance.
        stats (dict): A dictionary with 'mean' and 'std' keys for normalization.
    Returns:
        dict: A dictionary representing the normalized instance.
    """
    normalized_instance = {}
    for feature, value in instance.items():
        mean = stats["mean"].get(feature, 0)
        std = stats["std"].get(feature, 1)
        normalized_instance[feature] = (value - mean) / std
    return normalized_instance

@app.route(os.environ['AIP_HEALTH_ROUTE'], methods=['GET'])
def health_check():
    """Health check endpoint that returns the status of the server.
    Returns:
        Response: A Flask response with status 200 and "healthy" as the body.
    """
    return {"status": "healthy"}

@app.route(os.environ['AIP_PREDICT_ROUTE'], methods=['POST'])
def predict():
    """
    Prediction route that normalizes input data, and returns model predictions.
    Returns:
        Response: A Flask response containing JSON-formatted predictions.
    """
    request_json = request.get_json()
    request_instances = request_json['instances']

    # Normalize and format each instance
    formatted_instances = []
    for instance in request_instances:
        normalized_instance = normalize_data(instance, stats)
        formatted_instance = [
            normalized_instance['PT08.S1(CO)'],
            normalized_instance['NMHC(GT)'],
            normalized_instance['C6H6(GT)'],
            normalized_instance['PT08.S2(NMHC)'],
            normalized_instance['NOx(GT)'],
            normalized_instance['PT08.S3(NOx)'],
            normalized_instance['NO2(GT)'],
            normalized_instance['PT08.S4(NO2)'],
            normalized_instance['PT08.S5(O3)'],
            normalized_instance['T'],
            normalized_instance['RH'],
            normalized_instance['AH']
        ]
        formatted_instances.append(formatted_instance)

    # Make predictions with the model
    prediction = model.predict(formatted_instances)
    prediction = prediction.tolist()
    output = {'predictions': [{'result': pred} for pred in prediction]}
    return jsonify(output)

project_id, bucket_name = initialize_variables()
storage_client, bucket = initialize_client_and_bucket(bucket_name)
stats = load_stats(bucket)
model = load_model(bucket, bucket_name)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)