from google.cloud import storage
from datetime import datetime
import pytz
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import json
import gcsfs
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize variables
fs = gcsfs.GCSFileSystem()
storage_client = storage.Client()
bucket_name = os.getenv("BUCKET_NAME")
MODEL_DIR = os.environ['AIP_STORAGE_URI']

def load_data(gcs_train_data_path):
    """
    Loads the training data from Google Cloud Storage (GCS).
    
    Parameters:
    gcs_train_data_path (str): GCS path where the training data CSV is stored.
    
    Returns:
    DataFrame: A pandas DataFrame containing the training data.
    """
    with fs.open(gcs_train_data_path) as f:
        df = pd.read_parquet(f)
    
    return df

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

def data_transform(df):
    """
    Transforms the data by setting a datetime index, and splitting it into 
    training and validation sets. It also normalizes the features.
    
    Parameters:
    df (DataFrame): The DataFrame to be transformed.
    
    Returns:
    tuple: A tuple containing normalized training features, test features,
           normalized training labels, and test labels.
    """

    # Splitting the data into training and validation sets (80% training, 20% validation)
    X=df.drop('loan_status',axis=1)
    y=df['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)


     # Get the json from GCS
    client = storage.Client()
    bucket_name = os.getenv("BUCKET_NAME")
    blob_path = 'scaler/normalization_stats.json' # Change this to your blob path where the data is stored
    bucket = client.get_bucket("mlops_loan_data")
    blob = bucket.blob(blob_path)

    # Download the json as a string
    data = blob.download_as_string()
    stats = json.loads(data)

    # Normalize the data using the statistics from the training set
    X_train_scaled = normalize_data(X_train, stats)
    
    return X_train_scaled, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Trains a Random Forest Regressor model on the provided data.
    
    Parameters:
    X_train (DataFrame): The training features.
    y_train (Series): The training labels.
    
    Returns:
    RandomForestRegressor: The trained Random Forest model.
    """
    model = RandomForestClassifier(n_estimators=150, max_leaf_nodes=9, max_features=None, max_depth=9, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_and_upload_model(model, local_model_path, gcs_model_path):
    """
    Saves the model locally and uploads it to GCS.
    
    Parameters:
    model (RandomForestRegressor): The trained model to be saved and uploaded.
    local_model_path (str): The local path to save the model.
    gcs_model_path (str): The GCS path to upload the model.
    """
    # Save the model locally
    joblib.dump(model, local_model_path)

    # Upload the model to GCS
    with fs.open(gcs_model_path, 'wb') as f:
        joblib.dump(model, f)

def main():
    """
    Main function to orchestrate the loading of data, training of the model,
    and uploading the model to Google Cloud Storage.
    """
    # Load and transform data
    gcs_train_data_path = "gs://mlops_loan_data/data/train/train_data.csv"
    df = load_data(gcs_train_data_path)
    X_train, X_test, y_train, y_test = data_transform(df)

    # Train the model
    model = train_model(X_train, y_train)

    # Save the model locally and upload to GCS
    edt = pytz.timezone('US/Eastern')
    current_time_edt = datetime.now(edt)
    version = current_time_edt.strftime('%Y%m%d_%H%M%S')
    local_model_path = "model.pkl"
    gcs_model_path = f"{MODEL_DIR}/model_{version}.pkl"
    print(gcs_model_path)
    save_and_upload_model(model, local_model_path, gcs_model_path)

if __name__ == "__main__":
    main()