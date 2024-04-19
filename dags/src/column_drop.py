import logging
import os
import pickle

# Configure logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(PROJECT_DIR, 'logs', 'datapipeline.log')
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)  # Ensure the directory exists
logging.basicConfig(filename=LOG_PATH, level=logging.DEBUG, filemode='w', format=LOG_FORMAT)
logger = logging.getLogger(LOG_PATH)

INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_term.pkl')
OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_dropcol.pkl')

def drop_column(input_pickle_path=INPUT_PICKLE_PATH,
                            output_pickle_path=OUTPUT_PICKLE_PATH):

    """
    Function: drop_column

    Description:
    This function loads a pandas DataFrame from a specified pickle file, drops specific columns ('grade', 'title', 'emp_title') from the DataFrame, and saves the modified DataFrame back to a pickle file. If the input file path does not exist, it logs an error and raises FileNotFoundError.

    Parameters:
    - input_pickle_path (str): Path to the input pickle file containing the DataFrame. Default is INPUT_PICKLE_PATH.
    - output_pickle_path (str): Path to save the modified DataFrame as a pickle file. Default is OUTPUT_PICKLE_PATH.

    """
    logger.info(">>>>>>>>>>>>>>>>>>>>>> Started Column Drop Task <<<<<<<<<<<<<<<<<<<<<<")
    if os.path.exists(input_pickle_path):
        with open(input_pickle_path, "rb") as file:
            df = pickle.load(file)
    else:
        error_message = f"No data found at the specified path: {input_pickle_path}"
        logger.error(error_message)
        raise FileNotFoundError(error_message)

    df.drop(['sub_grade','title','emp_title'],axis=1,inplace=True)

    logger.info(">>>>>>>>>>>>>>>>>>>>>> Column Drop Task Completed <<<<<<<<<<<<<<<<<<<<<<")
    with open(output_pickle_path, "wb") as file:
        pickle.dump(df, file)
    logger.info(f"Data saved to {output_pickle_path}.")
    return output_pickle_path
