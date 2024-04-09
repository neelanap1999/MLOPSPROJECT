import os
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Configure logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(PROJECT_DIR, 'logs', 'datapipeline.log')
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)  # Ensure the directory exists
logging.basicConfig(filename='scaling_data.log', level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(LOG_PATH)

XTRAIN_INPUT_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'X_train.parquet')
XTEST_INPUT_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'X_test.parquet')


XTRAIN_OUTPUT_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'X_train_final.parquet')
XTEST_OUTPUT_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'X_test_final.parquet')

S_COLS = ['loan_amnt', 'int_rate', 'installment', 'annual_inc', 'dti', 
          'open_acc', 'pub_rec', 'revol_bal', 'total_acc', 'mort_acc', 'pub_rec_bankruptcies']

def scaler(xtrain_inpath=XTRAIN_INPUT_PATH, xtest_inpath=XTEST_INPUT_PATH, xtrain_outpath=XTRAIN_OUTPUT_PATH, xtest_outpath=XTEST_OUTPUT_PATH, cols_to_scale=S_COLS):
    
    logger.info(">>>>> Scaling Train and Test Data Started <<<<<<<<")
    if os.path.exists(xtrain_inpath) and os.path.exists(xtest_inpath):
        X_train=pd.read_parquet(xtrain_inpath)
        X_test=pd.read_parquet(xtest_inpath)
    else:
        error_message = f"No data found at the specified path: {xtrain_inpath} or {xtest_inpath}"
        logger.error(error_message)
        raise FileNotFoundError(error_message)

    cols_to_keep = [col for col in X_train.columns if col not in cols_to_scale]

    scaler_std = StandardScaler()
    X_train[cols_to_scale] = scaler_std.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler_std.transform(X_test[cols_to_scale])
    
    X_train.to_parquet(xtrain_outpath)
    X_test.to_parquet(xtest_outpath)
    logger.info(">>>>> Scaling Train and Test Data Completed <<<<<<<<")

    return (xtrain_outpath,xtest_outpath)