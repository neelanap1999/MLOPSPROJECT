import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from src.notification import notify_failure

# Configure logging
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_outlier.pkl')
OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_income_normalization.pkl')

# function to normalize the loan amount and annual income since they are skewed
def normalize_amount(input_pickle_path=INPUT_PICKLE_PATH,
                            output_pickle_path=OUTPUT_PICKLE_PATH):
    """
    Normalizing the 'loan_amnt' and 'annual_inc' columns of a DataFrame by taking the natural logarithm of their values.

    Args:
        input_pickle_path (str): The file path to the input pickle file containing the DataFrame.
                                Defaults to INPUT_PICKLE_PATH.
        output_pickle_path (str): The file path to save the output pickle file containing the normalized DataFrame.
                                 Defaults to OUTPUT_PICKLE_PATH.

    Returns:
        str: The file path where the normalized DataFrame is saved.
        
    Raises:
        FileNotFoundError: If no data is found at the specified input path.
    """

    if os.path.exists(input_pickle_path):
        with open(input_pickle_path, "rb") as file:
            df = pickle.load(file)
    else:
        raise FileNotFoundError(f"No data found at the specified path: {input_pickle_path}")
        notify_failure(f"No data found at the specified path: {input_pickle_path}")

    df['loan_amnt'] = np.log(df['loan_amnt'])
    df['annual_inc'] = np.log(df['annual_inc']) 

    with open(output_pickle_path, "wb") as file:
        pickle.dump(df, file)
    print(f"Data saved to {output_pickle_path}.")
    return output_pickle_path