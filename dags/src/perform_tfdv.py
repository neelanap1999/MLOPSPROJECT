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


def validate_data_tfdv(xtrain_inpath=XTRAIN_INPUT_PATH, xtest_inpath=XTEST_INPUT_PATH):

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

    # Infer schema from the computed statistics.
    X_train_schema = tfdv.infer_schema(statistics=X_train_stats)
    X_test_schema = tfdv.infer_schema(statistics=X_test_stats)

    # Display the inferred schema
    print(">>>>>>>>>>>>>> Train Schema <<<<<<<<<<<<<<<<<")
    tfdv.display_schema(X_train_schema)
    
    print(">>>>>>>>>>>>>> Test Schema <<<<<<<<<<<<<<<<<")
    tfdv.display_schema(X_test_schema)

    X_train_anomalies =  tfdv.validate_statistics(statistics=X_train_stats, schema=X_train_schema)
    X_test_anomalies =  tfdv.validate_statistics(statistics=X_test_stats, schema=X_test_schema)

    # Visualize anomalies
    tfdv.display_anomalies(X_train_anomalies)
    tfdv.display_anomalies(X_test_anomalies)


