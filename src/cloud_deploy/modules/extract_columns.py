# import logging
# import os
# import pickle

# Configure logging
# LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# LOG_PATH = os.path.join(PROJECT_DIR, 'logs', 'datapipeline.log')
# os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)  # Ensure the directory exists
# logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format=LOG_FORMAT)
# logger = logging.getLogger(LOG_PATH)

# INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_dropna.pkl')
# OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_year.pkl')

def extract_task(df):

    """
    Function: extract_year
    
    Description:
    This function loads a pandas DataFrame from a specified pickle file, extracts the year 
    from the 'earliest_cr_line' and 'issue_d' columns, and saves the modified DataFrame back 
    to a pickle file. If the input file path does not exist, it logs an error and raises FileNotFoundError.
    
    Parameters:
    - input_pickle_path (str): Path to the input pickle file containing the DataFrame. Default is INPUT_PICKLE_PATH.
    - output_pickle_path (str): Path to save the modified DataFrame as a pickle file. Default is OUTPUT_PICKLE_PATH.
    
    Returns:
    str: Path to the saved pickle file containing the modified DataFrame.
    """


    '''if os.path.exists(input_pickle_path):
        with open(input_pickle_path, "rb") as file:
            df = pickle.load(file)
    else:
        error_message = f"No data found at the specified path: {input_pickle_path}"
        logger.error(error_message)
        raise FileNotFoundError(error_message)'''

    df['earliest_cr_line']=df['earliest_cr_line'].apply(lambda x:int(x[-4:]))
    #df['issue_d']=df['issue_d'].apply(lambda x:int(x[0:4]))

    df['zipcode']=df['address'].apply(lambda x:str(x[-5:]))
    df["zipcode"] = df['zipcode'].astype ('category')
    df.drop('address',axis=1,inplace=True)

    # logger.info(f"Data saved to df after extracting year.")
    return df
