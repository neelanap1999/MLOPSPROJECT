import logging
import pandas as pd
from modules.extract_columns import extract_task
from modules.column_drop import drop_column
from modules.missing_values import handle_missing
from modules.null_drop import drop_null
from modules.dummies import get_dummies
from modules.outlier_handle import handle_outliers
from modules.labelencode import encode

# import os
import json 
import gcsfs

# Initialize a gcsfs file system object
fs = gcsfs.GCSFileSystem()

Stage_name = "Feature Extraction"

def stats_to_json(train_data, normalization_stats_gcs_path):
    
    Stage_name = "Preprocessing data"
    # logging.info(f"------Stage {Stage_name}------------")    
    Num_cols = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 
          'open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies']

    train_data_numeric = train_data[Num_cols]

    # Calculate mean and standard deviation for each feature in the training data
    mean_train = train_data_numeric.mean()
    std_train = train_data_numeric.std()

    # Store normalization statistics in a dictionary
    normalization_stats = {
        'mean': mean_train.to_dict(),
        'std': std_train.to_dict()
    }

    # Save the normalization statistics to a JSON file on GCS
    with fs.open(normalization_stats_gcs_path, 'w') as json_file:
        json.dump(normalization_stats, json_file)
    logging.info(f">>>>>> stage {Stage_name} completed <<<<<<\n\nx==========x")


def preprocess(train_data, output_filepath, normalization_stats_gcs_path):
    Stage_name = "Preprocessing data"
    logging.info(f"------Stage {Stage_name}------------")
    train_data = extract_task( train_data)
    train_data = encode(train_data)
    train_data = drop_column(train_data)
    train_data = handle_missing(train_data)
    train_data = drop_null(train_data)
    train_data = get_dummies(train_data)
    train_data = handle_outliers(train_data)
    

    #Generate stats on numerical columns only - >  export to JSON
    stats_to_json(train_data, normalization_stats_gcs_path)

    #save preprocessed data to output_file path
    with fs.open(output_filepath, 'w') as f:
            train_data.to_csv(output_filepath)
    logging.info(f">>>>>> stage {Stage_name} completed <<<<<<\n\nx==========x")


if __name__ == "__main__":
     
    #ouput gcs path to save train_data and test _data
    train_data_gcs_path = "gs://mlops_loan_data/data/train/train_data.xlsx"
    test_data_gcs_path = "gs://mlops_loan_data/data/train/test_data.xlsx"
    normalization_stats_gcs_path = "gs://mlops_loan_data/scaler/normalization_stats.json"
    preprocessed_train_data_gcs_path = "gs://mlops_loan_data/data/train/preprocess/train_data_preprocessed.csv"

    # Original dataset
    gcs_loan_data_path = "gs://mlops_loan_data/data/initial_data.csv"
    train_data = pd.read_csv(gcs_loan_data_path)

    preprocess(train_data, preprocessed_train_data_gcs_path, normalization_stats_gcs_path)


