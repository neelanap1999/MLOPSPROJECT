from flask import Flask, jsonify, request
from google.cloud import storage
import joblib
import os
import json
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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


def normalize_data(data, stats):
    """
    Normalizes the data using the provided statistics.

    Parameters:
    data (DataFrame): The data to be normalized.
    stats (dict): A dictionary containing the feature means and standard deviations.

    Returns:
    DataFrame: A pandas DataFrame containing the normalized data.
    """
    normalized_data = {}

    Num_cols = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 
          'open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies']
    
    for column in data.columns:
        if column not in Num_cols:
            mean = stats["mean"][column]
            std = stats["std"][column]
            normalized_data[column] = [(value - mean) / std for value in data[column]]
        else:
            # Keep categorical data unchanged
            normalized_data[column] = data[column]
    
    # Convert normalized_data dictionary back to a DataFrame
    normalized_df = pd.DataFrame(normalized_data, index=data.index)
    return normalized_df

def map_years(years):
    """
    Mapping employment length values to numerical values.

    Args:
        years (str or int): The employment length value.

    Returns:
        int or np.nan: The mapped numerical value for employment length, or np.nan if input is NaN.

    Raises:
        None
    """
    if pd.isna(years):  # Handle NaN values
        return np.nan
    elif isinstance(years, int):  # If already an integer, return as is
        return years
    elif years == '< 1 year':
        return 0
    elif years == '10+ years':
        return 10
    else:
        return int(years.split()[0])

def encode(df):

    dic = {' 36 months':36, ' 60 months':60}
    df['term'] = df.term.map(dic)

    # Applying mapping function to the column
    df['emp_length'] = df['emp_length'].map(map_years)

    # Label encoding for 'loan_status' column
    le=LabelEncoder()
    df['loan_status'] = le.fit_transform(df['loan_status'])

    # print(f"Data saved to df after encoding.")
    return df

def preprocess( df):
    df['earliest_cr_line']=df['earliest_cr_line'].apply(lambda x:int(x[-4:]))
    df['zipcode']=df['address'].apply(lambda x:str(x[-5:]))
    df.drop('address',axis=1,inplace=True)
    df = encode(df)
    df.drop(['sub_grade','title','emp_title'],axis=1,inplace=True)
    df['mort_acc'].fillna(df['mort_acc'].mean(),inplace = True)
    df['emp_length'].fillna(0,inplace = True)
    df['revol_util'].fillna(0,inplace = True)
    df['mort_acc'].fillna(0,inplace = True)
    df = pd.get_dummies(df,columns=['grade', 'verification_status', 'purpose', 'initial_list_status',
           'application_type', 'home_ownership'],dtype=int)   
    
    return df

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
    # instance = request_instances.to_dict()
    
    # Convert the dictionary to a list of dictionaries (Flask returns data as strings)
    # data = [instance]
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(request_instances)
    # df = preprocess(df)
    # df = normalize_data(df,stats)

    print(stats)
    print(df.columns)
    # Normalize and format each instance
    # formatted_instances = []
    # for instance in request_instances:
        # print(instance)
        # print(stats)
        # normalized_instance  = normalize_data(instance, stats)  
    #     formatted_instance = [
    #         normalized_instance['PT08.S1(CO)'],
    #         normalized_instance['NMHC(GT)'],
    #         normalized_instance['C6H6(GT)'],
    #         normalized_instance['PT08.S2(NMHC)'],
    #         normalized_instance['NOx(GT)'],
    #         normalized_instance['PT08.S3(NOx)'],
    #         normalized_instance['NO2(GT)'],
    #         normalized_instance['PT08.S4(NO2)'],
    #         normalized_instance['PT08.S5(O3)'],
    #         normalized_instance['T'],
    #         normalized_instance['RH'],
    #         normalized_instance['AH']
    #     ]
    #     formatted_instances.append(formatted_instance)

    # Make predictions with the model
    prediction = model.predict(df)
    prediction = prediction.tolist()
    output = {'predictions': [{'result': pred} for pred in prediction]}
    return jsonify(output)
    # return "recieve data"

project_id, bucket_name = initialize_variables()
storage_client, bucket = initialize_client_and_bucket(bucket_name)
stats = load_stats(bucket)
model = load_model(bucket, bucket_name)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)