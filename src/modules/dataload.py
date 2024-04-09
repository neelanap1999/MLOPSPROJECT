import logging
import pickle
import os
import pandas as pd

# Configure logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(PROJECT_DIR, 'logs', 'datapipeline.log')
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)  # Ensure the directory exists
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(LOG_PATH)

DEFAULT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data',
                                   'processed', 'initial.pkl')
DEFAULT_EXCEL_PATH = os.path.join(PROJECT_DIR, 'data', 'initial_data.csv')

def load_data(pickle_path=DEFAULT_PICKLE_PATH, excel_path=None):
  
    """
    Function: load_data
    
    Description:
    This function loads data from either a pickle file or an Excel file. It first attempts to 
    load the data from the specified pickle file path. If the pickle file does not exist, 
    it attempts to load the data from the specified Excel file path. If neither the pickle 
    nor the Excel file exists, it logs an error and raises FileNotFoundError. 
    After loading the data, it saves it to a pickle file for future use or re-saves it if loaded from an existing pickle file.
    
    Parameters:
    - pickle_path (str): Path to the pickle file containing the DataFrame. Default is DEFAULT_PICKLE_PATH.
    - excel_path (str or None): Path to the Excel file containing the DataFrame. If None, DEFAULT_EXCEL_PATH is used. Default is None.
    
    Returns:
    str: Path to the saved or re-saved pickle file containing the DataFrame.
    """

    logger.info("Loading data...")

    if excel_path is None:
        excel_path = DEFAULT_EXCEL_PATH
    df = None

    # If pickle doesn't exist, load from Excel
    if os.path.exists(excel_path):
        df = pd.read_csv(excel_path)
        logger.info(f"Data loaded from {excel_path}.")
    else:
        error_message = f"No data found in the specified paths: {excel_path}"
        logger.error(error_message)
        raise FileNotFoundError(error_message)
    # Save the data to pickle for future use (or re-save it if loaded from existing pickle)
    os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
    with open(pickle_path, "wb") as file:
        pickle.dump(df, file)
    logger.info(f"Data saved to {pickle_path} for future use.")
    return pickle_path


