import logging
import pandas as pd
#from utils.config import load_config
from modules.extract_columns import extract_task
from modules.column_drop import drop_column
from modules.missing_values import handle_missing
from modules.null_drop import drop_null
from modules.dummies import get_dummies
from modules.outlier_handle import handle_outliers
# from modules.income_normalization import normalize_amount
from modules.labelencode import encode
from modules.split import split
from modules.scaling_data import scaler
from modules.correlation import correlation
import os
import json 
import gcsfs

# Initialize a gcsfs file system object
fs = gcsfs.GCSFileSystem()

#config = load_config()
Stage_name = "Feature Extraction"

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#OUTPUT_PATH = os.path.join(PROJECT_DIR, 'data', 'feature_extract_out.parquet')

#JSON_FILE_PATH = os.path.join(PROJECT_DIR, 'data', 'stats_train.json')


def stats_to_json(df, json_file_path):
    
    S_COLS = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 
          'open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies']

    train_data_numeric = df[S_COLS]
    print(train_data_numeric)

    # Calculate mean and standard deviation for each feature in the training data
    mean_train = train_data_numeric.mean()
    std_train = train_data_numeric.std()

    # Store normalization statistics in a dictionary
    normalization_stats = {
        'mean': mean_train.to_dict(),
        'std': std_train.to_dict()
    }

    print(normalization_stats)

    # Writing the dictionary to a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(normalization_stats, json_file)


def preprocess(df_data, output_filepath, json_file_path):
    try:
        logging.info(f"------Stage {Stage_name}------------")
        #df_data = pd.read_csv("initial_data.csv")
        df_data = extract_task( df_data)
        df_data = encode(df_data)
        df_data = drop_column(df_data)
        df_data = handle_missing(df_data)
        df_data = drop_null(df_data)
        df_data = get_dummies(df_data)
        df_data = handle_outliers(df_data)
        df_data.to_parquet(output_filepath)
        stats_to_json(df_data, json_file_path)
        #Generate stats on numerical columns only - >  export to JSON

        logging.info(f">>>>>> stage {Stage_name} completed <<<<<<\n\nx==========x")
    except Exception as e:
            print(e)

if __name__ == "__main__":
     
     gcs_train_data_path = "gs://mlops_loan_data/data/AirQualityUCI.xlsx"
     normalization_stats_gcs_path = "gs://mlops_loan_data/scaler/normalization_stats.json"
     df_data = pd.read_csv(gcs_train_data_path)
     OUTPUT_PATH = "gs://mlops_loan_data/data/preprocess/data_preprocessed.csv"
     JSON_FILE_PATH = "gs://mlops_loan_data/preprocess/stats_train.json"
     preprocess(df_data, OUTPUT_PATH, JSON_FILE_PATH)


