import logging
import pickle
import os
import pandas as pd

# Configure logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'logs', 'dataload.log')
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)  # Ensure the directory exists
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(LOG_PATH)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DEFAULT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data',
                                   'processed', 'initial.pkl')
DEFAULT_EXCEL_PATH = os.path.join(PROJECT_DIR, 'data', 'initial.csv')

def load_data(pickle_path=DEFAULT_PICKLE_PATH, excel_path=None):
    logger.info("Loading data...")
    if excel_path is None:
        excel_path = DEFAULT_EXCEL_PATH
    df = None
    # Check if pickle file exists
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as file:
            df = pickle.load(file)
        logger.info(f"Data loaded successfully from {pickle_path}.")
    # If pickle doesn't exist, load from Excel
    elif os.path.exists(excel_path):
        df = pd.read_csv(excel_path)
        logger.info(f"Data loaded from {excel_path}.")
    else:
        error_message = f"No data found in the specified paths: {pickle_path} or {excel_path}"
        logger.error(error_message)
        raise FileNotFoundError(error_message)
    # Save the data to pickle for future use (or re-save it if loaded from existing pickle)
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    with open(pickle_path, "wb") as file:
        pickle.dump(df, file)
    logger.info(f"Data saved to {pickle_path} for future use.")
    return pickle_path


