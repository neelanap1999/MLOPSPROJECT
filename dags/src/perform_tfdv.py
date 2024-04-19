import tensorflow as tf
import tensorflow_data_validation as tfdv
import pandas as pd

from tensorflow_metadata.proto.v0 import schema_pb2
import pickle
import logging
import os

# Configure logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(PROJECT_DIR, 'logs', 'datapipeline.log')
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)  # Ensure the directory exists
logging.basicConfig(filename='tfdv.log', level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(LOG_PATH)

XTRAIN_INPUT_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'X_train.parquet')
XTEST_INPUT_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'X_test.parquet')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'data', 'processed') # Directory to store output files
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_stats(X_stats, filename):
    output_path = os.path.join(OUTPUT_DIR, filename)
    tfdv.write_stats_text(X_stats, output_path)
    logger.info(">>>>>>>>>>>>>>>> Saved Visualizations <<<<<<<<<<<<<<<<<<<")

def save_schema(schema, filename):
    output_path = os.path.join(OUTPUT_DIR, filename)
    with open(output_path, 'w') as f:
        f.write(str(schema))
    logger.info(">>>>>>>>>>>>>>>> Saved schema <<<<<<<<<<<<<<<<<<<")

def save_anomalies(anomalies, filename):
    output_path = os.path.join(OUTPUT_DIR, filename)
    with open(output_path, 'w') as f:
        f.write(str(anomalies))
    logger.info(">>>>>>>>>>>>>>>> Saved Anomalies <<<<<<<<<<<<<<<<<<<")

def validate_data_tfdv(xtrain_inpath=XTRAIN_INPUT_PATH, xtest_inpath=XTEST_INPUT_PATH):

    logger.info(">>>>>>>>>>>>>>>> Starting TFDV Data Validation <<<<<<<<<<<<<<<<<<<<<<")

    if os.path.exists(xtrain_inpath) and os.path.exists(xtest_inpath):
        X_train = pd.read_parquet(xtrain_inpath)
        X_test = pd.read_parquet(xtest_inpath)
    else:
        error_message = f"No data found at the specified path: {xtrain_inpath} or {xtest_inpath}"
        logger.error(error_message)
        raise FileNotFoundError(error_message)

    X_train_stats = tfdv.generate_statistics_from_dataframe(X_train)
    X_test_stats = tfdv.generate_statistics_from_dataframe(X_test)

    # Comparing Visualizing statistics of Train and Test
    tfdv.visualize_statistics(
            lhs_statistics=X_train_stats,  
            rhs_statistics=X_test_stats, 
            lhs_name="Train Data",
            rhs_name="Test Data"
        )
    save_stats(X_train_stats, 'train_stats.txt')
    save_stats(X_test_stats, 'test_stats.txt')

    # Infer schema from the computed statistics.
    X_train_schema = tfdv.infer_schema(statistics=X_train_stats)
    X_test_schema = tfdv.infer_schema(statistics=X_test_stats)

    tfdv.display_schema(schema=X_train_schema)
    tfdv.display_schema(schema=X_test_schema)

    # Save the inferred schemas
    save_schema(X_train_schema, 'train_schema.pbtxt')
    save_schema(X_test_schema, 'test_schema.pbtxt')

    # Display the inferred schema
    print(">>>>>>>>>>>>>> Train Schema <<<<<<<<<<<<<<<<<")
    print(X_train_schema)

    print(">>>>>>>>>>>>>> Test Schema <<<<<<<<<<<<<<<<<")
    print(X_test_schema)

    X_test_anomalies =  tfdv.validate_statistics(statistics=X_test_stats, schema=X_train_schema)

    # Save anomalies
    save_anomalies(X_test_anomalies, 'test_anomalies.pbtxt')

    # Visualize anomalies
    tfdv.display_anomalies(X_train_anomalies)
    tfdv.display_anomalies(X_test_anomalies)
    
    logger.info(">>>>>>>>>>>>>>>> TFDV Completed <<<<<<<<<<<<<<<<<<<")

