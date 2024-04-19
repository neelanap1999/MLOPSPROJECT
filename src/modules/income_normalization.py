import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Configure logging
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_outlier.pkl')
OUTPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'data', 'processed','after_income_normalization.pkl')

# function to normalize the loan amount and annual income since they are skewed
def normalize_amount(df):
    """
    Normalizing the 'loan_amnt' and 'annual_inc' columns of a DataFrame by taking the natural logarithm of their values.

    Args:
        input_pickle_path (str): The file path to the input pickle file containing the DataFrame.
                                Defaults to INPUT_PICKLE_PATH.
        output_pickle_path (str): The file path to save the output pickle file containing the normalized DataFrame.
                                 Defaults to OUTPUT_PICKLE_PATH.

    Returns:
        str: The file path where the normalized DataFrame is saved.
    """

    df['loan_amnt'] = np.log(df['loan_amnt'])
    df['annual_inc'] = np.log(df['annual_inc']) 

    print(f"Data saved to df after amount normailization.")
    return df