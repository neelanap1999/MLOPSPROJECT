import os
import pickle
import logging

# Configure logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(PROJECT_DIR, 'logs', 'datapipeline.log')
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)  # Ensure the directory exists
logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(LOG_PATH)


INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_fillna.pkl')
OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_dropna.pkl')

# function to drop null values
def drop_null(input_pickle_path=INPUT_PICKLE_PATH,
                            output_pickle_path=OUTPUT_PICKLE_PATH):
    """
    Droping the rows with missing values (NaN) from a DataFrame and saving the cleaned DataFrame to a pickle file.

    Args:
        input_pickle_path (str): The path to the input pickle file containing the DataFrame (Defaults to INPUT_PICKLE_PATH).
        output_pickle_path (str): The file path to save the output pickle file containing the cleaned DataFrame (Defaults to OUTPUT_PICKLE_PATH).

    Returns:
        str: The file path where the cleaned DataFrame is saved.
        
    Raises:
        FileNotFoundError: If no data is found at the specified input path.
    """

    if os.path.exists(input_pickle_path):
        with open(input_pickle_path, "rb") as file:
            df = pickle.load(file)
    else:
        error_message = f"No data found at the specified path: {input_pickle_path}"
        logger.error(error_message)
        raise FileNotFoundError(error_message)

    df.dropna(inplace=True)
    
    with open(output_pickle_path, "wb") as file:
        pickle.dump(df, file)
    logger.info(f"Data saved to {output_pickle_path}.")
    return output_pickle_path