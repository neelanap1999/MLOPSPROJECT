# import os
# import pickle
# import logging
# import pandas

# # Configure logging
# LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# LOG_PATH = os.path.join(PROJECT_DIR, 'logs', 'datapipeline.log')
# os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)  # Ensure the directory exists
# logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format=LOG_FORMAT)
# logger = logging.getLogger(LOG_PATH)

# INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_dropcol.pkl')
# OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_fillna.pkl')

def handle_missing(df):

    """
    -----
    Function: handle_missing
    
    Description:
    This function loads a pandas DataFrame from a specified pickle file, 
    handles missing values by filling them with appropriate values, and 
    saves the modified DataFrame back to a pickle file. If the input file path 
    does not exist, it logs an error and raises FileNotFoundError.
    
    Parameters:
    - input_pickle_path (str): Path to the input pickle file containing the DataFrame. Default is INPUT_PICKLE_PATH.
    - output_pickle_path (str): Path to save the modified DataFrame as a pickle file. Default is OUTPUT_PICKLE_PATH.
    
    Returns:
    str: Path to the saved pickle file containing the modified DataFrame.
    -----
    """


    df['mort_acc'].fillna(df['mort_acc'].mean(),inplace = True)
    df['emp_length'].fillna(0,inplace = True)
    df['revol_util'].fillna(0,inplace = True)
    df['mort_acc'].fillna(0,inplace = True)
    
    
    # logger.info(f"Data saved to df after removing missing values.")
    return df
