import os
import pickle
from sklearn.model_selection import train_test_split
import logging
import pandas as pd

# Configure logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(PROJECT_DIR, 'logs', 'datapipeline.log')
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)  # Ensure the directory exists
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(LOG_PATH)

INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','encoder_output.parquet')
XTRAIN_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','X_train.parquet')
XTEST_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','X_test.parquet')
YTRAIN_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','y_train.parquet')
YTEST_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','y_test.parquet')

def split(input_pickle_path=INPUT_PICKLE_PATH,
                            xtrain_path=XTRAIN_PATH, xtest_path=XTEST_PATH, ytrain_path=YTRAIN_PATH, ytest_path=YTEST_PATH ):


    df=pd.read_parquet(input_pickle_path)
    
    X=df.drop('loan_status',axis=1)
    y=df['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    X_train.to_parquet(xtrain_path)
    X_test.to_parquet(xtest_path)
    y_train.to_parquet(ytrain_path)
    y_test.to_parquet(ytest_path)
    
    return (xtrain_path,xtest_path,ytrain_path,ytest_path)